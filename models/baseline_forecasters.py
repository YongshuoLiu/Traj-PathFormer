import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_forecaster import (
    LearnableTimeEmbedding,
    PositionalEncoding,
    compute_motion_losses,
)


class AGDNGraphLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.att_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.msg_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_feat, valid_mask):
        bsz, lmax, _ = h.shape
        edge_h = self.edge_proj(edge_feat)
        src = h.unsqueeze(2).expand(-1, -1, lmax, -1)
        dst = h.unsqueeze(1).expand(-1, lmax, -1, -1)
        att_in = torch.cat([src, dst, edge_h], dim=-1)
        att = self.att_proj(att_in).squeeze(-1) / math.sqrt(self.hidden_dim)
        key_invalid = ~valid_mask.unsqueeze(1)
        att = att.masked_fill(key_invalid, -1e9)
        att = att.masked_fill(~valid_mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(att, dim=-1)
        msg_in = torch.cat([src, edge_h], dim=-1)
        msg = self.msg_proj(msg_in)
        agg = torch.einsum("bij,bijk->bik", weights, msg)
        out = self.out_proj(torch.cat([h, agg], dim=-1))
        out = self.norm(h + self.dropout(out))
        return out * valid_mask.unsqueeze(-1).float()


class TimeAwareModule(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(1, hidden_size)

    def forward(self, delta_t):
        decay = 1.0 / torch.log(torch.exp(torch.tensor(1.0, device=delta_t.device, dtype=delta_t.dtype)) + delta_t)
        return torch.sigmoid(self.fc(decay))


class TLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.time_module = TimeAwareModule(hidden_size)

    def forward(self, x_t, h_prev, c_prev, delta_t):
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev))
        c_tilde = torch.tanh(self.W_c(x_t) + self.U_c(h_prev))
        h_tilde = c_tilde + i_t
        h_short = torch.tanh(self.W_d(h_tilde))
        h_long_prev = h_tilde - h_short
        d_t = self.time_module(delta_t)
        h_short_weighted = h_short * d_t
        h_star = h_long_prev + h_short_weighted
        c_t = torch.tanh(h_star + o_t * c_prev)
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


@dataclass
class BaselineMotionConfig:
    model_name: str
    input_dim: int
    future_steps: int
    target_mode: str = "velocity"
    motion_dim: int = 2
    npatch: int = 8
    patch_len: int = 8
    hid_dim: int = 128
    te_dim: int = 16
    nlayer: int = 2
    nhead: int = 4
    tf_layer: int = 1
    motion_loss: str = "smoothl1"
    lambda_motion: float = 1.0
    lambda_traj: float = 1.0
    lambda_final: float = 1.0
    agdn_dropout: float = 0.0
    tlstm_layers: int = 1
    tlstm_dropout: float = 0.0


class BaselineForecastHead(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, future_steps: int, motion_dim: int):
        super().__init__()
        self.future_steps = int(future_steps)
        self.motion_dim = int(motion_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, future_steps * motion_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(x.size(0), self.future_steps, self.motion_dim)


class BaseFutureMotionForecaster(nn.Module):
    def __init__(self, cfg: BaselineMotionConfig):
        super().__init__()
        self.cfg = cfg
        self.forecast_head = BaselineForecastHead(
            in_dim=self.decoder_input_dim(),
            hid_dim=cfg.hid_dim,
            future_steps=cfg.future_steps,
            motion_dim=cfg.motion_dim,
        )

    def decoder_input_dim(self) -> int:
        raise NotImplementedError

    def encode_history(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_decoder_state(self, history: torch.Tensor, history_mask: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        valid_counts = history_mask.sum(dim=1).long().clamp(min=1, max=history.size(1))
        batch_idx = torch.arange(history.size(0), device=history.device)
        last_idx = valid_counts - 1
        last_obs = history[batch_idx, last_idx, :]
        last_hidden = encoded[batch_idx, last_idx, :]
        return torch.cat([last_obs, last_hidden], dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        history = batch["history"]
        history_mask = batch["history_mask"]
        encoded = self.encode_history(history, history_mask)
        decoder_state = self.get_decoder_state(history, history_mask, encoded)
        pred_motion_norm = self.forecast_head(decoder_state)
        return {
            "pred_motion_norm": torch.nan_to_num(pred_motion_norm, nan=0.0, posinf=1e4, neginf=-1e4),
            "encoded_history": encoded,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor], motion_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        position_scale = batch.get("position_scale")
        if position_scale is None:
            raise ValueError("batch must include position_scale.")
        out = self.forward(batch)
        pred_motion_norm = out["pred_motion_norm"]
        pred_motion_physical = pred_motion_norm * motion_scale.view(1, 1, -1)
        losses = compute_motion_losses(
            pred_motion_norm=pred_motion_norm,
            gt_motion_norm=batch["future_motion_norm"],
            pred_motion_physical=pred_motion_physical,
            gt_future_pos_norm=batch["future_pos_norm"],
            gt_future_pos=batch["future_pos"],
            future_dt=batch["future_dt"],
            position_scale=position_scale,
            target_mode=self.cfg.target_mode,
            lambda_motion=self.cfg.lambda_motion,
            lambda_traj=self.cfg.lambda_traj,
            lambda_final=self.cfg.lambda_final,
            loss_name=self.cfg.motion_loss,
        )
        return {
            **losses,
            "pred_motion_norm": pred_motion_norm,
            "pred_motion_physical": pred_motion_physical,
        }


class AGDNMotionForecaster(BaseFutureMotionForecaster):
    def __init__(self, cfg: BaselineMotionConfig):
        super().__init__(cfg)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hid_dim)
        self.layers = nn.ModuleList(
            [AGDNGraphLayer(cfg.hid_dim, edge_dim=4, dropout=cfg.agdn_dropout) for _ in range(cfg.nlayer)]
        )

    def decoder_input_dim(self) -> int:
        return self.cfg.input_dim + self.cfg.hid_dim

    def build_edge_features(self, history: torch.Tensor, ts_sequence: torch.Tensor) -> torch.Tensor:
        x = history
        lat = x[..., 0]
        lon = x[..., 1]
        src_flag = x[..., 6]

        dlat = lat.unsqueeze(2) - lat.unsqueeze(1)
        dlon = lon.unsqueeze(2) - lon.unsqueeze(1)
        dist = torch.sqrt(dlat.square() + dlon.square() + 1e-8)
        dt = (ts_sequence.unsqueeze(2) - ts_sequence.unsqueeze(1)).abs()
        same_src = (src_flag.unsqueeze(2) == src_flag.unsqueeze(1)).float()
        same_step = torch.eye(history.size(1), device=history.device, dtype=history.dtype).unsqueeze(0).expand(history.size(0), -1, -1)
        return torch.stack([dist, dt / 300.0, same_src, same_step], dim=-1)

    def encode_history(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = history_mask > 0
        ts_sequence = torch.cumsum(history[..., 5].clamp(min=0.0), dim=1)
        h = self.input_proj(history) * valid_mask.unsqueeze(-1).float()
        edge_feat = self.build_edge_features(history, ts_sequence)
        for layer in self.layers:
            h = layer(h, edge_feat, valid_mask)
        return h


class TLSTMMotionForecaster(BaseFutureMotionForecaster):
    def __init__(self, cfg: BaselineMotionConfig):
        super().__init__(cfg)
        self.input_proj = nn.Linear(cfg.input_dim, cfg.hid_dim)
        self.cells = nn.ModuleList(
            [TLSTMCell(cfg.hid_dim, cfg.hid_dim) for _ in range(max(int(cfg.tlstm_layers), 1))]
        )
        self.dropout_layer = nn.Dropout(cfg.tlstm_dropout)

    def decoder_input_dim(self) -> int:
        return self.cfg.input_dim + self.cfg.hid_dim

    def encode_history(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(history)
        bsz, lmax, _ = x.shape
        hs = [torch.zeros(bsz, self.cfg.hid_dim, device=x.device, dtype=x.dtype) for _ in self.cells]
        cs = [torch.zeros(bsz, self.cfg.hid_dim, device=x.device, dtype=x.dtype) for _ in self.cells]
        outputs = []
        delta_sequence = history[..., 5].clamp(min=0.0)

        for t in range(lmax):
            step_in = x[:, t, :]
            delta_t = delta_sequence[:, t].unsqueeze(-1).to(dtype=x.dtype)
            step_valid = history_mask[:, t].unsqueeze(-1)
            for layer_idx, cell in enumerate(self.cells):
                h_new, c_new = cell(step_in, hs[layer_idx], cs[layer_idx], delta_t)
                hs[layer_idx] = step_valid * h_new + (1.0 - step_valid) * hs[layer_idx]
                cs[layer_idx] = step_valid * c_new + (1.0 - step_valid) * cs[layer_idx]
                step_in = hs[layer_idx]
                if layer_idx < len(self.cells) - 1:
                    step_in = self.dropout_layer(step_in)
            outputs.append(step_in.unsqueeze(1))
        return torch.cat(outputs, dim=1) * history_mask.unsqueeze(-1)


class TPatchMotionForecaster(BaseFutureMotionForecaster):
    def __init__(self, cfg: BaselineMotionConfig):
        super().__init__(cfg)
        self.time_embed = LearnableTimeEmbedding(cfg.te_dim, mode="learnable")
        self.point_dim = cfg.input_dim + cfg.te_dim
        self.ttcn_dim = cfg.hid_dim - 1
        self.filter_generators = nn.Sequential(
            nn.Linear(self.point_dim, self.ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.point_dim * self.ttcn_dim, bias=True),
        )
        self.t_bias = nn.Parameter(torch.randn(1, self.ttcn_dim))
        self.pos_encoding = PositionalEncoding(cfg.hid_dim)
        self.transformer_encoder = nn.ModuleList()
        for _ in range(cfg.nlayer):
            enc_layer = nn.TransformerEncoderLayer(d_model=cfg.hid_dim, nhead=cfg.nhead, batch_first=True)
            self.transformer_encoder.append(
                nn.TransformerEncoder(enc_layer, num_layers=cfg.tf_layer, enable_nested_tensor=False)
            )

    def decoder_input_dim(self) -> int:
        return self.cfg.input_dim + self.cfg.hid_dim

    def patchify(self, history: torch.Tensor, history_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, lmax, dim = history.shape
        lmax_expected = self.cfg.npatch * self.cfg.patch_len
        if lmax != lmax_expected:
            if lmax < lmax_expected:
                pad = lmax_expected - lmax
                history = F.pad(history, (0, 0, 0, pad))
                history_mask = F.pad(history_mask, (0, pad))
            else:
                history = history[:, :lmax_expected, :]
                history_mask = history_mask[:, :lmax_expected]
        hist_time = torch.cumsum(history[..., 5].clamp(min=0.0), dim=1)
        return {
            "x_patch": history.view(bsz, self.cfg.npatch, self.cfg.patch_len, dim),
            "ts_patch": hist_time.view(bsz, self.cfg.npatch, self.cfg.patch_len, 1),
            "mask_pt": history_mask.view(bsz, self.cfg.npatch, self.cfg.patch_len, 1),
        }

    def ttcn(self, x_int: torch.Tensor, mask_x: torch.Tensor) -> torch.Tensor:
        n_items, lmax, _ = mask_x.shape
        filters = self.filter_generators(x_int)
        filters = filters * mask_x + (1.0 - mask_x) * (-1e8)
        filters = torch.softmax(filters, dim=-2)
        filters = filters.view(n_items, lmax, self.ttcn_dim, self.point_dim)
        x_broad = x_int.unsqueeze(-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(x_broad * filters, dim=-3), dim=-1)
        return torch.relu(ttcn_out + self.t_bias)

    def encode_history(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        pack = self.patchify(history, history_mask)
        te = self.time_embed(pack["ts_patch"])
        x_int = torch.cat([pack["x_patch"], te], dim=-1)
        bsz = history.size(0)
        x_int_flat = x_int.view(bsz * self.cfg.npatch, self.cfg.patch_len, self.point_dim)
        mask_flat = pack["mask_pt"].view(bsz * self.cfg.npatch, self.cfg.patch_len, 1)
        patch_feat = self.ttcn(x_int_flat, mask_flat)
        patch_exists = (mask_flat.sum(dim=1) > 0).float()
        patch_tokens = torch.cat([patch_feat, patch_exists], dim=-1).view(bsz, self.cfg.npatch, self.cfg.hid_dim)
        patch_mask = (pack["mask_pt"].sum(dim=2).squeeze(-1) > 0).float()
        h = patch_tokens * patch_mask.unsqueeze(-1)
        src_key_padding_mask = patch_mask <= 0
        for encoder in self.transformer_encoder:
            h = self.pos_encoding(h)
            h = encoder(h, src_key_padding_mask=src_key_padding_mask)
            h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
            h = h * patch_mask.unsqueeze(-1)

        patch_counts = torch.clamp(patch_mask.sum(dim=1).long(), min=1, max=self.cfg.npatch)
        last_patch_idx = patch_counts - 1
        batch_idx = torch.arange(bsz, device=history.device)
        last_patch_hidden = h[batch_idx, last_patch_idx, :]
        seq_hidden = last_patch_hidden.unsqueeze(1).expand(-1, history.size(1), -1)
        return seq_hidden * history_mask.unsqueeze(-1)


class LightweightNamedForecaster(BaseFutureMotionForecaster):
    """Small baseline proxies with paper-name-compatible interfaces.

    These modules are intentionally lightweight. They provide runnable baselines
    for quick comparisons when the original full implementations are not present
    in the repository.
    """

    def __init__(self, cfg: BaselineMotionConfig):
        super().__init__(cfg)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
        )
        self.time_embed = LearnableTimeEmbedding(cfg.te_dim, mode="learnable")
        self.time_proj = nn.Linear(cfg.hid_dim + cfg.te_dim, cfg.hid_dim)
        self.rnn = nn.GRU(cfg.hid_dim, cfg.hid_dim, num_layers=1, batch_first=True)
        self.conv3 = nn.Conv1d(cfg.hid_dim, cfg.hid_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(cfg.hid_dim, cfg.hid_dim, kernel_size=5, padding=2)
        self.gate_conv = nn.Conv1d(cfg.hid_dim, cfg.hid_dim * 2, kernel_size=3, padding=1)
        self.patch_proj = nn.Linear(cfg.patch_len * cfg.input_dim, cfg.hid_dim)
        self.pos_encoding = PositionalEncoding(cfg.hid_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hid_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.hid_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=max(int(cfg.tf_layer), 1), enable_nested_tensor=False)
        self.mix = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
        )
        self.decay_proj = nn.Linear(1, cfg.hid_dim)
        self.query_proj = nn.Linear(cfg.te_dim, cfg.hid_dim)
        self.attn = nn.MultiheadAttention(cfg.hid_dim, cfg.nhead, batch_first=True)
        self.norm = nn.LayerNorm(cfg.hid_dim)

    def decoder_input_dim(self) -> int:
        return self.cfg.input_dim + self.cfg.hid_dim

    def _with_time(self, history: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ts = torch.cumsum(history[..., 5].clamp(min=0.0), dim=1).unsqueeze(-1)
        te = self.time_embed(ts)
        return torch.relu(self.time_proj(torch.cat([h, te], dim=-1)))

    def _conv_encode(self, h: torch.Tensor, history_mask: torch.Tensor, gated: bool = False) -> torch.Tensor:
        z = h.transpose(1, 2)
        if gated:
            a, b = self.gate_conv(z).chunk(2, dim=1)
            y = torch.tanh(a) * torch.sigmoid(b)
        else:
            y = torch.relu(self.conv3(z)) + torch.relu(self.conv5(z))
        y = y.transpose(1, 2)
        return self.norm(h + y) * history_mask.unsqueeze(-1)

    def _rnn_encode(self, h: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        lengths = history_mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=h.size(1))
        return out * history_mask.unsqueeze(-1)

    def _transformer_encode(self, h: torch.Tensor, history_mask: torch.Tensor, use_time: bool = False, history: torch.Tensor = None) -> torch.Tensor:
        if use_time and history is not None:
            h = self._with_time(history, h)
        h = self.pos_encoding(h)
        key_padding_mask = history_mask <= 0
        out = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4) * history_mask.unsqueeze(-1)

    def _patch_encode(self, history: torch.Tensor, history_mask: torch.Tensor, cross: bool = False) -> torch.Tensor:
        bsz, lmax, dim = history.shape
        need = self.cfg.npatch * self.cfg.patch_len
        if lmax < need:
            history = F.pad(history, (0, 0, 0, need - lmax))
            history_mask = F.pad(history_mask, (0, need - lmax))
        elif lmax > need:
            history = history[:, :need]
            history_mask = history_mask[:, :need]
        x = history.view(bsz, self.cfg.npatch, self.cfg.patch_len * dim)
        patch_mask = (history_mask.view(bsz, self.cfg.npatch, self.cfg.patch_len).sum(dim=-1) > 0).float()
        p = self.patch_proj(x) * patch_mask.unsqueeze(-1)
        if cross:
            p = self.norm(p + self.mix(p))
        p = self.pos_encoding(p)
        p = self.transformer(p, src_key_padding_mask=patch_mask <= 0)
        p = torch.nan_to_num(p, nan=0.0, posinf=1e4, neginf=-1e4) * patch_mask.unsqueeze(-1)
        patch_counts = patch_mask.sum(dim=1).long().clamp(min=1, max=self.cfg.npatch)
        last_patch = p[torch.arange(bsz, device=history.device), patch_counts - 1]
        return last_patch.unsqueeze(1).expand(-1, lmax, -1)[:, : history_mask.size(1)] * history_mask.unsqueeze(-1)

    def _spectral_encode(self, h: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.rfft(h * history_mask.unsqueeze(-1), dim=1)
        keep = max(2, min(freq.size(1), 6))
        filt = torch.zeros_like(freq)
        filt[:, :keep] = freq[:, :keep]
        y = torch.fft.irfft(filt, n=h.size(1), dim=1)
        return self.norm(h + self.mix(y)) * history_mask.unsqueeze(-1)

    def _attention_pool_encode(self, history: torch.Tensor, h: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        ts = torch.cumsum(history[..., 5].clamp(min=0.0), dim=1).unsqueeze(-1)
        q = self.query_proj(self.time_embed(ts))
        out, _ = self.attn(q, h, h, key_padding_mask=history_mask <= 0)
        return self.norm(h + out) * history_mask.unsqueeze(-1)

    def encode_history(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        name = self.cfg.model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
        h = self.input_proj(history) * history_mask.unsqueeze(-1)

        if name == "dlinear":
            trend = F.avg_pool1d(h.transpose(1, 2), kernel_size=5, stride=1, padding=2).transpose(1, 2)
            seasonal = h - trend
            return self.norm(self.mix(trend) + seasonal) * history_mask.unsqueeze(-1)
        if name == "timesnet":
            return self._conv_encode(h, history_mask, gated=False)
        if name == "patchtst":
            return self._patch_encode(history, history_mask, cross=False)
        if name == "crossformer":
            return self._patch_encode(history, history_mask, cross=True)
        if name in {"graphwavenet", "mtgnn"}:
            return self._conv_encode(h, history_mask, gated=True)
        if name in {"stemgnn", "fouriergnn"}:
            return self._spectral_encode(h, history_mask)
        if name == "crossgnn":
            graph = torch.softmax(torch.matmul(h, h.transpose(1, 2)) / math.sqrt(self.cfg.hid_dim), dim=-1)
            graph = graph.masked_fill((history_mask <= 0).unsqueeze(1), 0.0)
            return self.norm(h + torch.matmul(graph, self.mix(h))) * history_mask.unsqueeze(-1)
        if name == "grud":
            dt = history[..., 5:6].clamp(min=0.0)
            decay = torch.exp(-torch.relu(self.decay_proj(dt)))
            return self._rnn_encode(h * decay, history_mask)
        if name in {"seft", "raindrop", "warpformer", "mtand"}:
            return self._attention_pool_encode(history, self._with_time(history, h), history_mask)
        if name in {"latentode", "cru", "neuralflow"}:
            z = self._rnn_encode(h, history_mask)
            dt = history[..., 5:6].clamp(min=0.0)
            flow = torch.tanh(self.mix(z)) * torch.log1p(dt)
            if name == "cru":
                gate = torch.sigmoid(self.decay_proj(dt))
                return (gate * z + (1.0 - gate) * flow) * history_mask.unsqueeze(-1)
            return self.norm(z + flow) * history_mask.unsqueeze(-1)
        return self._rnn_encode(h, history_mask)


def build_baseline_model(cfg: BaselineMotionConfig) -> BaseFutureMotionForecaster:
    name = cfg.model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if name == "agdn":
        return AGDNMotionForecaster(cfg)
    if name == "tlstm":
        return TLSTMMotionForecaster(cfg)
    if name in {"tpatch", "tpatchgnn"}:
        return TPatchMotionForecaster(cfg)
    if name in {
        "dlinear",
        "timesnet",
        "patchtst",
        "crossformer",
        "graphwavenet",
        "mtgnn",
        "stemgnn",
        "crossgnn",
        "fouriergnn",
        "grud",
        "seft",
        "raindrop",
        "warpformer",
        "mtand",
        "latentode",
        "cru",
        "neuralflow",
    }:
        cfg.model_name = name
        return LightweightNamedForecaster(cfg)
    raise ValueError(f"Unsupported baseline model: {cfg.model_name}")
