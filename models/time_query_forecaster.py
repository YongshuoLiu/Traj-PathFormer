import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .patch_forecaster import (
    BaseMotionPatchModel,
    LearnableTimeEmbedding,
    MotionPatchConfig,
    compute_motion_losses,
)


@dataclass
class TimeQueryMotionPatchConfig:
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
    motion_loss: str = "smoothl1"
    lambda_motion: float = 1.0
    lambda_traj: float = 1.0
    lambda_final: float = 1.0
    future_te_dim: int = 16
    decoder_dropout: float = 0.1
    use_kinematic_prior: bool = True
    max_future_dt_for_embed: float = 600.0
    prior_min_dt: float = 1.0

    def to_backbone_config(self) -> MotionPatchConfig:
        return MotionPatchConfig(
            input_dim=self.input_dim,
            future_steps=self.future_steps,
            target_mode=self.target_mode,
            motion_dim=self.motion_dim,
            npatch=self.npatch,
            patch_len=self.patch_len,
            hid_dim=self.hid_dim,
            te_dim=self.te_dim,
            nlayer=self.nlayer,
            nhead=self.nhead,
            tf_layer=self.tf_layer,
            time_embedding_mode=self.time_embedding_mode,
            intra_patch_encoder=self.intra_patch_encoder,
            use_exists_bias=self.use_exists_bias,
            use_patch_gattn=self.use_patch_gattn,
            use_transformer=self.use_transformer,
            use_positional_encoding=self.use_positional_encoding,
            tau_seconds=self.tau_seconds,
            motion_loss=self.motion_loss,
            lambda_motion=self.lambda_motion,
            lambda_traj=self.lambda_traj,
            lambda_final=self.lambda_final,
            teacher_mode="shared",
            lambda_recovery=0.0,
            lambda_completion=0.0,
        )


class MotionPatchTimeQueryForecaster(BaseMotionPatchModel):
    def __init__(self, cfg: TimeQueryMotionPatchConfig):
        super().__init__(cfg.to_backbone_config())
        self.query_cfg = cfg
        self.register_buffer("motion_scale_buffer", torch.ones(cfg.motion_dim, dtype=torch.float32))
        self.future_time_embed = LearnableTimeEmbedding(cfg.future_te_dim, mode="learnable")
        self.future_step_embed = nn.Embedding(cfg.future_steps, cfg.hid_dim)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(cfg.input_dim + cfg.hid_dim),
            nn.Linear(cfg.input_dim + cfg.hid_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim + cfg.future_te_dim + cfg.hid_dim),
            nn.Linear(cfg.hid_dim + cfg.future_te_dim + cfg.hid_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.decoder_dropout),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
        )
        self.future_cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hid_dim,
            num_heads=cfg.nhead,
            batch_first=True,
            dropout=cfg.decoder_dropout,
        )
        self.decoder_norm = nn.LayerNorm(cfg.hid_dim)
        self.step_decoder = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim * 2 + cfg.input_dim + cfg.future_te_dim),
            nn.Linear(cfg.hid_dim * 2 + cfg.input_dim + cfg.future_te_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.decoder_dropout),
            nn.Linear(cfg.hid_dim, cfg.motion_dim),
        )
        self.prior_gate = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim * 2 + cfg.input_dim + cfg.future_te_dim),
            nn.Linear(cfg.hid_dim * 2 + cfg.input_dim + cfg.future_te_dim, cfg.motion_dim),
        )

    def _extract_last_valid_obs(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        valid_counts = history_mask.sum(dim=1).long().clamp(min=1, max=history.size(1))
        batch_idx = torch.arange(history.size(0), device=history.device)
        return history[batch_idx, valid_counts - 1]

    def _future_time_feature(self, future_dt: torch.Tensor) -> torch.Tensor:
        denom = math.log1p(max(float(self.query_cfg.max_future_dt_for_embed), 1.0))
        return torch.log1p(future_dt.clamp(min=0.0)).div(denom).unsqueeze(-1)

    def _last_velocity_prior(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        history_raw = batch.get("history_raw")
        if history_raw is None or self.query_cfg.target_mode != "velocity":
            return torch.zeros(
                batch["history"].size(0),
                self.query_cfg.motion_dim,
                device=batch["history"].device,
                dtype=batch["history"].dtype,
            )

        history_mask = batch["history_mask"] > 0
        xy = history_raw[..., : self.query_cfg.motion_dim]
        dt = history_raw[..., 5].clamp(min=float(self.query_cfg.prior_min_dt))
        step_velocity = torch.zeros_like(xy)
        step_velocity[:, 1:, :] = (xy[:, 1:, :] - xy[:, :-1, :]) / dt[:, 1:].unsqueeze(-1)
        step_velocity = torch.nan_to_num(step_velocity, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-200.0, max=200.0)

        prev_valid = torch.zeros_like(history_mask)
        prev_valid[:, 1:] = history_mask[:, :-1]
        valid_step = history_mask & prev_valid & (history_raw[..., 5] >= float(self.query_cfg.prior_min_dt))
        recency = torch.linspace(0.2, 1.0, history_raw.size(1), device=history_raw.device, dtype=history_raw.dtype)
        weights = valid_step.float() * recency.unsqueeze(0)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        prior_velocity = (step_velocity * weights.unsqueeze(-1)).sum(dim=1) / denom
        motion_scale = self.motion_scale_buffer.to(device=prior_velocity.device, dtype=prior_velocity.dtype).view(1, -1)
        return prior_velocity / motion_scale.clamp(min=1e-6)

    def _build_future_queries(
        self,
        last_obs: torch.Tensor,
        contextual_tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        future_dt: torch.Tensor,
    ) -> torch.Tensor:
        patch_counts = torch.clamp(patch_mask.sum(dim=1).long(), min=1, max=self.cfg.npatch)
        last_patch_idx = patch_counts - 1
        batch_idx = torch.arange(last_obs.size(0), device=last_obs.device)
        last_ctx = contextual_tokens[batch_idx, last_patch_idx, :]
        global_token = self.global_proj(torch.cat([last_obs, last_ctx], dim=-1))

        dt_feat = self._future_time_feature(future_dt)
        dt_embed = self.future_time_embed(dt_feat)
        step_ids = torch.arange(self.query_cfg.future_steps, device=last_obs.device)
        step_embed = self.future_step_embed(step_ids).unsqueeze(0).expand(last_obs.size(0), -1, -1)
        global_expand = global_token.unsqueeze(1).expand(-1, self.query_cfg.future_steps, -1)
        query_in = torch.cat([global_expand, dt_embed, step_embed], dim=-1)
        return self.query_proj(query_in)

    def set_motion_scale(self, motion_scale: torch.Tensor) -> None:
        motion_scale = motion_scale.detach().to(
            device=self.motion_scale_buffer.device,
            dtype=self.motion_scale_buffer.dtype,
        )
        self.motion_scale_buffer.copy_(motion_scale)

    def forward(self, return_loss: bool = False, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pack = self.patchify(batch["history"], batch["history_mask"])
        encoded = self.backbone(
            pack["x_patch"],
            pack["ts_patch"],
            pack["mask_pt"],
            pack["patch_mask"],
            forced_mask=None,
            apply_mask=False,
        )
        last_obs = self._extract_last_valid_obs(batch["history"], batch["history_mask"])
        queries = self._build_future_queries(last_obs, encoded["contextual_tokens"], pack["patch_mask"], batch["future_dt"])
        attended, attn_weights = self.future_cross_attn(
            query=queries,
            key=encoded["contextual_tokens"],
            value=encoded["contextual_tokens"],
            key_padding_mask=pack["patch_mask"] <= 0,
            need_weights=True,
        )
        attended = self.decoder_norm(attended + queries)
        dt_feat = self._future_time_feature(batch["future_dt"])
        dt_embed = self.future_time_embed(dt_feat)
        last_obs_expand = last_obs.unsqueeze(1).expand(-1, self.query_cfg.future_steps, -1)
        decoder_in = torch.cat([attended, queries, last_obs_expand, dt_embed], dim=-1)
        pred_motion_norm = self.step_decoder(decoder_in)
        if self.query_cfg.use_kinematic_prior:
            prior = self._last_velocity_prior(batch).unsqueeze(1)
            pred_motion_norm = pred_motion_norm + torch.sigmoid(self.prior_gate(decoder_in)) * prior
        pred_motion_norm = torch.nan_to_num(pred_motion_norm, nan=0.0, posinf=1e4, neginf=-1e4)
        out = {
            "pred_motion_norm": pred_motion_norm,
            "future_queries": queries,
            "future_tokens": attended,
            "attn_weights": attn_weights,
            "patch_mask": pack["patch_mask"],
        }
        if not return_loss:
            return out

        position_scale = batch.get("position_scale")
        if position_scale is None:
            raise ValueError("batch must include position_scale for normalized trajectory loss.")
        pred_motion_physical = pred_motion_norm * self.motion_scale_buffer.view(1, 1, -1)
        losses = compute_motion_losses(
            pred_motion_norm=pred_motion_norm,
            gt_motion_norm=batch["future_motion_norm"],
            pred_motion_physical=pred_motion_physical,
            gt_future_pos_norm=batch["future_pos_norm"],
            gt_future_pos=batch["future_pos"],
            future_dt=batch["future_dt"],
            position_scale=position_scale,
            target_mode=self.query_cfg.target_mode,
            lambda_motion=self.query_cfg.lambda_motion,
            lambda_traj=self.query_cfg.lambda_traj,
            lambda_final=self.query_cfg.lambda_final,
            loss_name=self.query_cfg.motion_loss,
        )
        return {
            **losses,
            "pred_motion_norm": pred_motion_norm,
            "pred_motion_physical": pred_motion_physical,
            "patch_mask": out["patch_mask"],
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor], motion_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.set_motion_scale(motion_scale)
        return self.forward(return_loss=True, **batch)
