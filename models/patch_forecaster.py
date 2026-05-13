import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MotionPatchConfig:
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
    time_embedding_mode: str = "learnable"
    intra_patch_encoder: str = "gat"
    use_exists_bias: bool = True
    use_patch_gattn: bool = True
    use_transformer: bool = True
    use_positional_encoding: bool = True
    tau_seconds: float = 300.0
    patch_mask_strategy: str = "token"
    decoder_context_mode: str = "contextual"
    point_decoder_mode: str = "mlp"
    motion_loss: str = "smoothl1"
    lambda_motion: float = 1.0
    lambda_traj: float = 1.0
    lambda_final: float = 1.0
    teacher_mode: str = "ema"
    teacher_momentum: float = 0.999
    recovery_target_level: str = "contextual"
    recovery_loss: str = "mse"
    lambda_recovery: float = 1.0
    completion_target_level: str = "local"
    completion_loss: str = "mse"
    lambda_completion: float = 0.0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LearnableTimeEmbedding(nn.Module):
    def __init__(self, te_dim: int, mode: str = "learnable"):
        super().__init__()
        self.te_dim = int(te_dim)
        self.mode = mode
        if self.te_dim <= 0:
            raise ValueError("te_dim must be positive.")
        if mode == "learnable":
            periodic_dim = max(self.te_dim - 1, 0)
            self.scale = nn.Linear(1, 1)
            self.periodic = nn.Linear(1, periodic_dim) if periodic_dim > 0 else None
        elif mode == "linear":
            self.linear = nn.Linear(1, self.te_dim)
        elif mode == "none":
            self.register_parameter("dummy", None)
        else:
            raise ValueError(f"Unsupported time_embedding_mode: {mode}")

    def forward(self, tt: torch.Tensor) -> torch.Tensor:
        if self.mode == "learnable":
            out1 = self.scale(tt)
            if self.periodic is None:
                return out1
            out2 = torch.sin(self.periodic(tt))
            return torch.cat([out1, out2], dim=-1)
        if self.mode == "linear":
            return self.linear(tt)
        return torch.zeros(*tt.shape[:-1], self.te_dim, device=tt.device, dtype=tt.dtype)


class PatchGraphAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, tau_seconds: float = 300.0, delta_seconds: float = 300.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.tau = float(tau_seconds)
        self.delta = float(delta_seconds)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _time_bias(self, m: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(m, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().float() * self.delta
        w = torch.exp(-dist / max(self.tau, 1e-6))
        return torch.log(w + 1e-12)

    def forward(self, h: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, num_patch, dim = h.shape
        q = self.q_proj(h).view(bsz, num_patch, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(bsz, num_patch, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(bsz, num_patch, self.nhead, self.d_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn + self._time_bias(num_patch, h.device)[None, None, :, :]
        if patch_mask is not None:
            key_pad = (~patch_mask) if patch_mask.dtype == torch.bool else (patch_mask <= 0)
            attn = attn.masked_fill(key_pad[:, None, None, :], -1e9)
        weights = torch.softmax(attn, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, num_patch, dim)
        out = self.out_proj(out)
        if patch_mask is not None:
            out = out * patch_mask.unsqueeze(-1).float()
        return out


class SimplePatchMapping(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.net(x)
        if patch_mask is not None:
            out = out * patch_mask.unsqueeze(-1).float()
        return out


class IntraPatchPointGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, nhead, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.squeeze(-1) > 0
        h = self.in_proj(x)
        key_padding_mask = ~valid
        has_any = valid.any(dim=1)
        if (~has_any).any():
            empty_rows = torch.where(~has_any)[0]
            key_padding_mask[empty_rows, 0] = False
            h[empty_rows, 0, :] = 0.0
        z, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        h = self.norm1(h + z)
        h = self.norm2(h + self.ffn(h))
        w = valid.float().unsqueeze(-1)
        denom = w.sum(dim=1).clamp(min=1.0)
        return (h * w).sum(dim=1) / denom


class TTCNPatchEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if out_dim < 2:
            raise ValueError("out_dim must be at least 2.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ttcn_dim = out_dim - 1
        self.filter_generator = nn.Sequential(
            nn.Linear(in_dim, self.ttcn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, self.ttcn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.ttcn_dim, in_dim * self.ttcn_dim),
        )
        self.bias = nn.Parameter(torch.randn(1, self.ttcn_dim))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_items, lmax, _ = x.shape
        filters = self.filter_generator(x)
        filters = filters * mask + (1.0 - mask) * (-1e8)
        filters = torch.softmax(filters, dim=-2)
        filters = torch.nan_to_num(filters, nan=0.0, posinf=0.0, neginf=0.0)
        filters = filters.view(n_items, lmax, self.ttcn_dim, self.in_dim)
        x_broad = x.unsqueeze(-2).repeat(1, 1, self.ttcn_dim, 1)
        ttcn_out = torch.sum(torch.sum(x_broad * filters, dim=-3), dim=-1)
        patch_feat = torch.relu(ttcn_out + self.bias)
        patch_feat = torch.nan_to_num(patch_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        patch_exists = (mask.sum(dim=1) > 0).float()
        patch_token = torch.cat([patch_feat, patch_exists], dim=-1)
        return self.norm(patch_token)


def build_patch_projector(mode: str, dim: int) -> nn.Module:
    if mode == "identity":
        return nn.Identity()
    if mode == "linear":
        return nn.Linear(dim, dim)
    if mode == "mlp":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
    raise ValueError(f"Unsupported projector mode: {mode}")


def compute_patch_loss(pred: torch.Tensor, target: torch.Tensor, select_mask: torch.Tensor, loss_type: str) -> torch.Tensor:
    select_mask = select_mask.bool()
    if not select_mask.any():
        return pred.sum() * 0.0
    pred_sel = pred[select_mask]
    tgt_sel = target[select_mask]
    if loss_type == "mse":
        return F.mse_loss(pred_sel, tgt_sel, reduction="mean")
    if loss_type == "smoothl1":
        return F.smooth_l1_loss(pred_sel, tgt_sel, reduction="mean")
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(pred_sel, tgt_sel, dim=-1)).mean()
    raise ValueError(f"Unsupported loss type: {loss_type}")


def reconstruct_positions(pred_motion_physical: torch.Tensor, future_dt: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "velocity":
        step_delta = pred_motion_physical * future_dt.unsqueeze(-1)
    elif target_mode == "displacement":
        step_delta = pred_motion_physical
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return torch.cumsum(step_delta, dim=1)


def compute_motion_losses(
    pred_motion_norm: torch.Tensor,
    gt_motion_norm: torch.Tensor,
    pred_motion_physical: torch.Tensor,
    gt_future_pos_norm: torch.Tensor,
    gt_future_pos: torch.Tensor,
    future_dt: torch.Tensor,
    position_scale: torch.Tensor,
    target_mode: str,
    lambda_motion: float,
    lambda_traj: float,
    lambda_final: float,
    loss_name: str,
) -> Dict[str, torch.Tensor]:
    if loss_name == "mse":
        point_loss = F.mse_loss
    elif loss_name == "smoothl1":
        point_loss = F.smooth_l1_loss
    else:
        raise ValueError(f"Unsupported motion loss: {loss_name}")
    pred_future_pos = reconstruct_positions(pred_motion_physical, future_dt, target_mode)
    pred_future_pos_norm = pred_future_pos / position_scale.view(1, 1, -1)
    motion_loss = point_loss(pred_motion_norm, gt_motion_norm, reduction="mean")
    traj_loss = point_loss(pred_future_pos_norm, gt_future_pos_norm, reduction="mean")
    final_loss = point_loss(pred_future_pos_norm[:, -1], gt_future_pos_norm[:, -1], reduction="mean")
    total = lambda_motion * motion_loss + lambda_traj * traj_loss + lambda_final * final_loss
    with torch.no_grad():
        pos_mae = (pred_future_pos - gt_future_pos).abs().mean()
        final_mae = (pred_future_pos[:, -1] - gt_future_pos[:, -1]).abs().mean()
    return {
        "loss": total,
        "motion_loss": motion_loss,
        "traj_loss": traj_loss,
        "final_loss": final_loss,
        "pred_future_pos": pred_future_pos,
        "pos_mae": pos_mae,
        "final_mae": final_mae,
    }


class PatchBackbone(nn.Module):
    def __init__(self, cfg: MotionPatchConfig):
        super().__init__()
        self.cfg = cfg
        self.hid_dim = int(cfg.hid_dim)
        self.te_dim = int(cfg.te_dim)
        self.npatch = int(cfg.npatch)
        self.patch_len = int(cfg.patch_len)
        self.n_layer = int(cfg.nlayer)
        self.nhead = int(cfg.nhead)
        self.tf_layer = int(cfg.tf_layer)
        self.in_dim = int(cfg.input_dim)
        self.point_dim = self.in_dim + self.te_dim

        self.time_embed = LearnableTimeEmbedding(self.te_dim, mode=cfg.time_embedding_mode)
        if cfg.intra_patch_encoder == "gat":
            self.intra_patch_encoder = IntraPatchPointGAT(self.point_dim, self.hid_dim, nhead=self.nhead)
        elif cfg.intra_patch_encoder == "ttcn":
            self.intra_patch_encoder = TTCNPatchEncoder(self.point_dim, self.hid_dim)
        else:
            raise ValueError(f"Unsupported intra_patch_encoder: {cfg.intra_patch_encoder}")

        self.use_exists_bias = bool(cfg.use_exists_bias)
        self.exists_bias = nn.Parameter(torch.zeros(1, 1, self.hid_dim))
        self.use_patch_gattn = bool(cfg.use_patch_gattn)
        self.use_transformer = bool(cfg.use_transformer)
        self.use_positional_encoding = bool(cfg.use_positional_encoding)
        self.patch_gattn = nn.ModuleList()
        self.patch_gattn_fallback = nn.ModuleList()
        self.transformer_encoder = nn.ModuleList()
        self.transformer_fallback = nn.ModuleList()
        for _ in range(self.n_layer):
            self.patch_gattn.append(PatchGraphAttention(self.hid_dim, nhead=self.nhead, tau_seconds=cfg.tau_seconds))
            self.patch_gattn_fallback.append(SimplePatchMapping(self.hid_dim))
            enc_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=self.nhead, batch_first=True)
            self.transformer_encoder.append(nn.TransformerEncoder(enc_layer, num_layers=self.tf_layer, enable_nested_tensor=False))
            self.transformer_fallback.append(SimplePatchMapping(self.hid_dim))
        self.pos_encoding = PositionalEncoding(self.hid_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.hid_dim))

    def build_point_features(self, x_patch: torch.Tensor, ts_patch: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(ts_patch)
        return torch.cat([x_patch, te], dim=-1)

    def encode_local(self, x_patch: torch.Tensor, ts_patch: torch.Tensor, mask_pt: torch.Tensor):
        bsz = x_patch.size(0)
        point_feat = self.build_point_features(x_patch, ts_patch)
        point_feat_flat = point_feat.view(bsz * self.npatch, self.patch_len, self.point_dim)
        mask_flat = mask_pt.view(bsz * self.npatch, self.patch_len, 1)
        local = self.intra_patch_encoder(point_feat_flat, mask_flat)
        local = torch.nan_to_num(local, nan=0.0, posinf=1e4, neginf=-1e4)
        local = local.view(bsz, self.npatch, self.hid_dim)
        patch_exists = (mask_flat.sum(dim=1) > 0).float().view(bsz, self.npatch, 1)
        if self.use_exists_bias:
            local = local + patch_exists * self.exists_bias
        return point_feat, local, patch_exists

    def apply_patch_mask(self, local_tokens: torch.Tensor, patch_mask: torch.Tensor, forced_mask: Optional[torch.Tensor]):
        valid_mask = patch_mask > 0
        if forced_mask is None:
            return local_tokens, torch.zeros_like(valid_mask, dtype=torch.bool)
        masked = forced_mask.bool() & valid_mask
        if not masked.any():
            return local_tokens, masked
        x = local_tokens.clone()
        if self.cfg.patch_mask_strategy == "token":
            x[masked] = self.mask_token.expand_as(x)[masked]
        elif self.cfg.patch_mask_strategy == "zero":
            x[masked] = 0.0
        else:
            raise ValueError(f"Unsupported patch_mask_strategy: {self.cfg.patch_mask_strategy}")
        return x, masked

    def contextualize(self, tokens: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        x = tokens
        key_padding_mask = patch_mask <= 0
        for layer in range(self.n_layer):
            if self.use_patch_gattn:
                x = x + self.patch_gattn[layer](x, patch_mask)
            else:
                x = x + self.patch_gattn_fallback[layer](x, patch_mask)
            x_in = self.pos_encoding(x) if self.use_positional_encoding else x
            if self.use_transformer:
                x = x + self.transformer_encoder[layer](x_in, src_key_padding_mask=key_padding_mask)
            else:
                x = x + self.transformer_fallback[layer](x_in, patch_mask)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = x * patch_mask.unsqueeze(-1).float()
        return x

    def forward(
        self,
        x_patch: torch.Tensor,
        ts_patch: torch.Tensor,
        mask_pt: torch.Tensor,
        patch_mask: torch.Tensor,
        forced_mask: Optional[torch.Tensor] = None,
        apply_mask: bool = False,
    ) -> Dict[str, torch.Tensor]:
        point_feat, local_tokens, patch_exists = self.encode_local(x_patch, ts_patch, mask_pt)
        masked_tokens = local_tokens
        applied_mask = torch.zeros_like(patch_mask, dtype=torch.bool)
        if apply_mask:
            masked_tokens, applied_mask = self.apply_patch_mask(local_tokens, patch_mask, forced_mask)
        contextual_tokens = self.contextualize(masked_tokens, patch_mask)
        return {
            "point_feat": point_feat,
            "local_tokens": local_tokens,
            "masked_tokens": masked_tokens,
            "contextual_tokens": contextual_tokens,
            "patch_exists": patch_exists,
            "applied_mask": applied_mask,
        }


class BaseMotionPatchModel(nn.Module):
    def __init__(self, cfg: MotionPatchConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = PatchBackbone(cfg)
        self.motion_head = nn.Sequential(
            nn.LayerNorm(cfg.input_dim + cfg.hid_dim),
            nn.Linear(cfg.input_dim + cfg.hid_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hid_dim, cfg.future_steps * cfg.motion_dim),
        )

    def patchify(self, history: torch.Tensor, history_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, lmax, dim = history.shape
        assert dim == self.cfg.input_dim
        npatch = self.cfg.npatch
        patch_len = self.cfg.patch_len
        lmax_expected = npatch * patch_len
        if lmax != lmax_expected:
            if lmax < lmax_expected:
                pad = lmax_expected - lmax
                history = F.pad(history, (0, 0, 0, pad))
                history_mask = F.pad(history_mask, (0, pad))
            else:
                history = history[:, :lmax_expected, :]
                history_mask = history_mask[:, :lmax_expected]
        hist_time = torch.cumsum(history[..., 5].clamp(min=0.0), dim=1)
        x_patch = history.view(bsz, npatch, patch_len, dim)
        ts_patch = hist_time.view(bsz, npatch, patch_len, 1)
        mask_pt = history_mask.view(bsz, npatch, patch_len, 1)
        patch_mask = (mask_pt.sum(dim=2).squeeze(-1) > 0).float()
        return {
            "x_patch": x_patch,
            "ts_patch": ts_patch,
            "mask_pt": mask_pt,
            "patch_mask": patch_mask,
        }

    def _sample_one_patch_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        valid = patch_mask > 0
        sampled = torch.zeros_like(valid, dtype=torch.bool)
        for b in range(valid.size(0)):
            valid_idx = torch.where(valid[b])[0]
            if valid_idx.numel() <= 1:
                continue
            pick = valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=patch_mask.device)]
            sampled[b, pick] = True
        return sampled

    def _last_context(
        self,
        history: torch.Tensor,
        contextual_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        point_mask = history_mask if history_mask is not None else patch_mask.repeat_interleave(self.cfg.patch_len, dim=1)
        valid_counts = point_mask.sum(dim=1).long().clamp(min=1, max=history.size(1))
        batch_idx = torch.arange(history.size(0), device=history.device)
        last_obs = history[batch_idx, valid_counts - 1, :]
        patch_counts = torch.clamp(patch_mask.sum(dim=1).long(), min=1, max=self.cfg.npatch)
        last_patch_idx = patch_counts - 1
        last_ctx = contextual_tokens[batch_idx, last_patch_idx, :]
        return torch.cat([last_obs, last_ctx], dim=-1)

    def _predict_motion(self, decoder_in: torch.Tensor) -> torch.Tensor:
        pred = self.motion_head(decoder_in)
        pred = pred.view(decoder_in.size(0), self.cfg.future_steps, self.cfg.motion_dim)
        return torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)


class MotionPatchPretrainForecaster(BaseMotionPatchModel):
    """
    Pretraining / no-completion version.
    Uses the patch backbone and a future-motion head only.
    """

    def __init__(self, cfg: MotionPatchConfig):
        super().__init__(cfg)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pack = self.patchify(batch["history"], batch["history_mask"])
        student = self.backbone(
            pack["x_patch"],
            pack["ts_patch"],
            pack["mask_pt"],
            pack["patch_mask"],
            forced_mask=None,
            apply_mask=False,
        )
        decoder_in = self._last_context(
            batch["history"], student["contextual_tokens"], pack["patch_mask"], batch["history_mask"]
        )
        pred_motion_norm = self._predict_motion(decoder_in)
        return {
            "pred_motion_norm": pred_motion_norm,
            "student_features": student,
            "patch_mask": pack["patch_mask"],
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor], motion_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        position_scale = batch["position_scale"] if "position_scale" in batch else None
        if position_scale is None:
            raise ValueError("batch must include position_scale for normalized position losses.")
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
            "patch_mask": out["patch_mask"],
        }


class MotionPatchCompletionForecaster(BaseMotionPatchModel):
    """
    Completion / recovery version.
    Student sees one masked patch, teacher sees the full patch sequence.
    Future motion prediction stays as the main forecasting head.
    """

    def __init__(self, cfg: MotionPatchConfig):
        super().__init__(cfg)
        if cfg.teacher_mode not in {"ema", "shared"}:
            raise ValueError("Completion forecaster expects teacher_mode in {'ema', 'shared'}.")
        if cfg.teacher_mode == "ema":
            self.teacher_backbone = copy.deepcopy(self.backbone)
            for param in self.teacher_backbone.parameters():
                param.requires_grad = False
        else:
            self.teacher_backbone = None
        self.teacher_projector = build_patch_projector("identity", cfg.hid_dim)
        self.completion_head = build_patch_projector("mlp", cfg.hid_dim)

    def update_teacher(self) -> None:
        if self.cfg.teacher_mode != "ema":
            return
        with torch.no_grad():
            student_state = dict(self.backbone.named_parameters())
            for name, teacher_param in self.teacher_backbone.named_parameters():
                teacher_param.data.mul_(self.cfg.teacher_momentum).add_(
                    student_state[name].data, alpha=1.0 - self.cfg.teacher_momentum
                )
            student_buffers = dict(self.backbone.named_buffers())
            for name, teacher_buffer in self.teacher_backbone.named_buffers():
                teacher_buffer.data.copy_(student_buffers[name].data)

    def _run_teacher(
        self,
        x_patch: torch.Tensor,
        ts_patch: torch.Tensor,
        mask_pt: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.cfg.teacher_mode == "shared":
            train_state = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                out = self.backbone(x_patch, ts_patch, mask_pt, patch_mask, forced_mask=None, apply_mask=False)
            if train_state:
                self.backbone.train()
        else:
            with torch.no_grad():
                self.teacher_backbone.eval()
                out = self.teacher_backbone(x_patch, ts_patch, mask_pt, patch_mask, forced_mask=None, apply_mask=False)
        return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in out.items()}

    def _select_patch_tensor(self, features: Dict[str, torch.Tensor], level: str) -> torch.Tensor:
        if level == "contextual":
            return features["contextual_tokens"]
        if level == "local":
            return features["local_tokens"]
        if level == "masked":
            return features["masked_tokens"]
        raise ValueError(f"Unsupported level: {level}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pack = self.patchify(batch["history"], batch["history_mask"])
        sampled_mask = self._sample_one_patch_mask(pack["patch_mask"])
        student = self.backbone(
            pack["x_patch"],
            pack["ts_patch"],
            pack["mask_pt"],
            pack["patch_mask"],
            forced_mask=sampled_mask,
            apply_mask=True,
        )
        teacher = self._run_teacher(pack["x_patch"], pack["ts_patch"], pack["mask_pt"], pack["patch_mask"])
        decoder_in = self._last_context(
            batch["history"], student["contextual_tokens"], pack["patch_mask"], batch["history_mask"]
        )
        pred_motion_norm = self._predict_motion(decoder_in)
        return {
            "pred_motion_norm": pred_motion_norm,
            "student_features": student,
            "teacher_features": teacher,
            "sampled_patch_mask": sampled_mask,
            "patch_mask": pack["patch_mask"],
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor], motion_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        position_scale = batch["position_scale"] if "position_scale" in batch else None
        if position_scale is None:
            raise ValueError("batch must include position_scale for normalized position losses.")
        out = self.forward(batch)
        pred_motion_norm = out["pred_motion_norm"]
        pred_motion_physical = pred_motion_norm * motion_scale.view(1, 1, -1)
        motion_losses = compute_motion_losses(
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
        student = out["student_features"]
        teacher = out["teacher_features"]
        sampled_mask = out["sampled_patch_mask"]

        recovery_loss = compute_patch_loss(
            self._select_patch_tensor(student, self.cfg.recovery_target_level),
            self.teacher_projector(self._select_patch_tensor(teacher, self.cfg.recovery_target_level)),
            sampled_mask,
            self.cfg.recovery_loss,
        )
        total = motion_losses["loss"] + self.cfg.lambda_recovery * recovery_loss

        completion_loss = pred_motion_norm.sum() * 0.0
        if self.cfg.lambda_completion > 0:
            completion_pred = self.completion_head(student["contextual_tokens"])
            completion_target = self._select_patch_tensor(teacher, self.cfg.completion_target_level)
            completion_loss = compute_patch_loss(
                completion_pred,
                completion_target,
                sampled_mask,
                self.cfg.completion_loss,
            )
            total = total + self.cfg.lambda_completion * completion_loss

        return {
            **motion_losses,
            "loss": total,
            "recovery_loss": recovery_loss,
            "completion_loss": completion_loss,
            "pred_motion_norm": pred_motion_norm,
            "pred_motion_physical": pred_motion_physical,
            "patch_mask": out["patch_mask"],
            "sampled_patch_mask": sampled_mask,
        }
