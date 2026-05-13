import argparse
import json
import math
import os
import random
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise SystemExit("PyTorch is required for training but is not installed in the current environment.") from exc

from models.patch_forecaster import (
    IntraPatchPointGAT,
    LearnableTimeEmbedding,
    PatchGraphAttention,
    PositionalEncoding,
    compute_motion_losses,
)
from pair_patch_data import (
    FEAT_DT,
    PairPatchDataset,
    append_log,
    build_experiment_dir,
    build_or_load_cache,
    log_device_info,
    move_batch,
    resolve_device,
)


@dataclass
class CrossCompletionConfig:
    input_dim: int = 6
    npatch: int = 8
    max_points_per_patch: int = 16
    future_steps: int = 12
    target_mode: str = "velocity"
    hid_dim: int = 128
    te_dim: int = 16
    future_te_dim: int = 16
    nhead: int = 4
    tf_layer: int = 2
    patch_minutes: int = 15
    tau_seconds: float = 300.0
    decoder_dropout: float = 0.1
    motion_loss: str = "smoothl1"
    lambda_motion: float = 1.0
    lambda_traj: float = 1.0
    lambda_final: float = 1.0
    lambda_completion: float = 1.0
    mask_patch_count: int = 1
    local_chunk_size: int = 3
    max_future_dt_for_embed: float = 600.0
    sparse_training_prob: float = 0.5
    sparse_keep_points: int = 2


class PairPatchCrossCompletionModel(nn.Module):
    def __init__(self, cfg: CrossCompletionConfig):
        super().__init__()
        self.cfg = cfg
        self.point_dim = cfg.input_dim + cfg.te_dim
        self.time_embed = LearnableTimeEmbedding(cfg.te_dim, mode="learnable")
        self.patch_encoder = IntraPatchPointGAT(self.point_dim, cfg.hid_dim, nhead=cfg.nhead, dropout=0.1)
        self.patch_time_proj = nn.Sequential(
            nn.Linear(3, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
        )
        self.query_time_mlp = nn.Sequential(
            nn.Linear(3, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
        )
        self.patch_index_embed = nn.Embedding(cfg.npatch, cfg.hid_dim)
        self.completion_query_mlp = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim * 6),
            nn.Linear(cfg.hid_dim * 6, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
        )
        self.key_mlp = nn.Linear(cfg.hid_dim, cfg.hid_dim)
        self.value_mlp = nn.Linear(cfg.hid_dim, cfg.hid_dim)
        self.completion_norm = nn.LayerNorm(cfg.hid_dim)
        self.completion_gate = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim * 3),
            nn.Linear(cfg.hid_dim * 3, cfg.hid_dim),
            nn.Sigmoid(),
        )
        self.pos_encoding = PositionalEncoding(cfg.hid_dim)
        self.anchor_patch_attn = PatchGraphAttention(
            cfg.hid_dim,
            nhead=cfg.nhead,
            tau_seconds=cfg.tau_seconds,
            delta_seconds=float(cfg.patch_minutes) * 60.0,
        )
        self.sender_patch_attn = PatchGraphAttention(
            cfg.hid_dim,
            nhead=cfg.nhead,
            tau_seconds=cfg.tau_seconds,
            delta_seconds=float(cfg.patch_minutes) * 60.0,
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=cfg.hid_dim, nhead=cfg.nhead, batch_first=True)
        self.anchor_transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.tf_layer, enable_nested_tensor=False)
        self.sender_transformer = nn.TransformerEncoder(enc_layer, num_layers=cfg.tf_layer, enable_nested_tensor=False)
        self.future_time_embed = LearnableTimeEmbedding(cfg.future_te_dim, mode="learnable")
        self.future_step_embed = nn.Embedding(cfg.future_steps, cfg.hid_dim)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim + cfg.input_dim),
            nn.Linear(cfg.hid_dim + cfg.input_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
        )
        self.future_query_proj = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim + cfg.future_te_dim + cfg.hid_dim),
            nn.Linear(cfg.hid_dim + cfg.future_te_dim + cfg.hid_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.decoder_dropout),
            nn.Linear(cfg.hid_dim, cfg.hid_dim),
        )
        self.future_cross_attn = nn.MultiheadAttention(
            cfg.hid_dim,
            cfg.nhead,
            batch_first=True,
            dropout=cfg.decoder_dropout,
        )
        self.future_decoder_norm = nn.LayerNorm(cfg.hid_dim)
        self.step_decoder = nn.Sequential(
            nn.LayerNorm(cfg.hid_dim * 2 + cfg.input_dim + cfg.future_te_dim),
            nn.Linear(cfg.hid_dim * 2 + cfg.input_dim + cfg.future_te_dim, cfg.hid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.decoder_dropout),
            nn.Linear(cfg.hid_dim, 2),
        )
        self.register_buffer("motion_scale_buffer", torch.ones(2, dtype=torch.float32))

    def set_motion_scale(self, motion_scale: torch.Tensor) -> None:
        motion_scale = motion_scale.detach().to(
            device=self.motion_scale_buffer.device,
            dtype=self.motion_scale_buffer.dtype,
        )
        self.motion_scale_buffer.copy_(motion_scale)

    def encode_stream(self, patch_x: torch.Tensor, patch_mask: torch.Tensor, patch_hms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, npatch, max_points, _ = patch_x.shape
        tt = patch_x[..., FEAT_DT : FEAT_DT + 1]
        point_feat = torch.cat([patch_x, self.time_embed(tt)], dim=-1)
        point_feat = point_feat.view(bsz * npatch, max_points, self.point_dim)
        mask_flat = patch_mask.view(bsz * npatch, max_points, 1)
        local = self.patch_encoder(point_feat, mask_flat).view(bsz, npatch, self.cfg.hid_dim)
        patch_valid = (patch_mask.sum(dim=-1) > 0).float()
        local = local + self.patch_time_proj(patch_hms)
        return local * patch_valid.unsqueeze(-1), patch_valid

    def sample_masked_patches(self, anchor_valid: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(anchor_valid, dtype=torch.bool)
        for b in range(anchor_valid.size(0)):
            valid_idx = torch.where(anchor_valid[b] > 0)[0]
            if valid_idx.numel() <= 1:
                continue
            valid_idx = valid_idx[:-1]
            if valid_idx.numel() <= 0:
                continue
            count = min(self.cfg.mask_patch_count, int(valid_idx.numel()))
            perm = torch.randperm(valid_idx.numel(), device=anchor_valid.device)[:count]
            masked[b, valid_idx[perm]] = True
        return masked

    def apply_patch_mask(
        self,
        anchor_patch: torch.Tensor,
        anchor_mask: torch.Tensor,
        masked_patch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masked_pt = masked_patch.unsqueeze(-1).expand(-1, -1, anchor_mask.size(-1))
        masked_pt = masked_pt.to(dtype=anchor_mask.dtype)
        keep_patch = (1.0 - masked_patch.float()).unsqueeze(-1).unsqueeze(-1)
        patch_x = anchor_patch * keep_patch
        patch_mask = anchor_mask * (1.0 - masked_pt)
        return patch_x, patch_mask

    def apply_sparse_patch_mask(
        self,
        anchor_patch: torch.Tensor,
        anchor_mask: torch.Tensor,
        masked_patch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sparse_mask = anchor_mask.clone()
        keep_points = max(int(self.cfg.sparse_keep_points), 1)
        for b in range(anchor_mask.size(0)):
            patch_ids = torch.where(masked_patch[b])[0]
            for p in patch_ids.tolist():
                valid = torch.where(anchor_mask[b, p] > 0)[0]
                if valid.numel() <= keep_points:
                    continue
                select_pos = torch.linspace(0, valid.numel() - 1, keep_points, device=valid.device).round().long()
                keep_idx = valid[select_pos].unique()
                new_mask = torch.zeros_like(anchor_mask[b, p])
                new_mask[keep_idx] = 1.0
                sparse_mask[b, p] = new_mask
        return anchor_patch * sparse_mask.unsqueeze(-1), sparse_mask

    def apply_training_corruption(
        self,
        anchor_patch: torch.Tensor,
        anchor_mask: torch.Tensor,
        masked_patch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.cfg.sparse_training_prob > 0:
            if torch.rand((), device=anchor_patch.device) < float(self.cfg.sparse_training_prob):
                return self.apply_sparse_patch_mask(anchor_patch, anchor_mask, masked_patch)
        return self.apply_patch_mask(anchor_patch, anchor_mask, masked_patch)

    def contextualize(self, x: torch.Tensor, valid: torch.Tensor, encoder: nn.Module, patch_attn: nn.Module) -> torch.Tensor:
        x = x + patch_attn(x, valid)
        x = self.pos_encoding(x)
        key_padding_mask = valid <= 0
        x = x + encoder(x, src_key_padding_mask=key_padding_mask)
        return x * valid.unsqueeze(-1)

    def select_local_chunk(
        self,
        sender_ctx: torch.Tensor,
        sender_valid: torch.Tensor,
        anchor_patch_start: torch.Tensor,
        sender_patch_start: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_size = int(self.cfg.local_chunk_size)
        start_dist = (anchor_patch_start.unsqueeze(-1) - sender_patch_start.unsqueeze(1)).abs()
        valid_mask = sender_valid > 0
        start_dist = start_dist.masked_fill(~valid_mask.unsqueeze(1), float("inf"))
        nearest_idx = torch.topk(start_dist, k=chunk_size, dim=-1, largest=False).indices

        gathered_start = torch.gather(
            sender_patch_start.unsqueeze(1).expand(-1, anchor_patch_start.size(1), -1),
            2,
            nearest_idx,
        )
        sort_order = gathered_start.argsort(dim=-1)
        nearest_idx = torch.gather(nearest_idx, 2, sort_order)

        gather_idx = nearest_idx.unsqueeze(-1).expand(-1, -1, -1, sender_ctx.size(-1))
        chunk_ctx = torch.gather(
            sender_ctx.unsqueeze(1).expand(-1, anchor_patch_start.size(1), -1, -1),
            2,
            gather_idx,
        )
        chunk_valid = torch.gather(
            valid_mask.unsqueeze(1).expand(-1, anchor_patch_start.size(1), -1),
            2,
            nearest_idx,
        )
        return chunk_ctx, chunk_valid

    def summarize_local_chunk(self, chunk_ctx: torch.Tensor, chunk_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_f = chunk_valid.unsqueeze(-1).float()
        denom = valid_f.sum(dim=2).clamp(min=1.0)
        summary = (chunk_ctx * valid_f).sum(dim=2) / denom

        left_idx = chunk_valid.float().argmax(dim=2)
        rev_valid = torch.flip(chunk_valid, dims=[2]).float()
        right_rev_idx = rev_valid.argmax(dim=2)
        right_idx = chunk_valid.size(2) - 1 - right_rev_idx

        left = torch.gather(
            chunk_ctx,
            2,
            left_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, chunk_ctx.size(-1)),
        ).squeeze(2)
        right = torch.gather(
            chunk_ctx,
            2,
            right_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, chunk_ctx.size(-1)),
        ).squeeze(2)

        has_any = chunk_valid.any(dim=2, keepdim=True)
        left = torch.where(has_any, left, torch.zeros_like(left))
        right = torch.where(has_any, right, torch.zeros_like(right))
        summary = torch.where(has_any, summary, torch.zeros_like(summary))
        return left, right, summary

    def build_completion_query(
        self,
        masked_anchor_local: torch.Tensor,
        anchor_patch_hms: torch.Tensor,
        chunk_ctx: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> torch.Tensor:
        bsz, npatch, _ = anchor_patch_hms.shape
        time_q = self.query_time_mlp(anchor_patch_hms)
        patch_ids = torch.arange(npatch, device=anchor_patch_hms.device).unsqueeze(0).expand(bsz, -1)
        idx_q = self.patch_index_embed(patch_ids)
        left_ctx, right_ctx, summary_ctx = self.summarize_local_chunk(chunk_ctx, chunk_valid)
        return self.completion_query_mlp(
            torch.cat([masked_anchor_local, time_q, idx_q, left_ctx, right_ctx, summary_ctx], dim=-1)
        )

    def complete_anchor(
        self,
        masked_anchor_local: torch.Tensor,
        sender_ctx: torch.Tensor,
        sender_valid: torch.Tensor,
        masked_patch: torch.Tensor,
        anchor_patch_hms: torch.Tensor,
        anchor_patch_start: torch.Tensor,
        sender_patch_start: torch.Tensor,
    ) -> torch.Tensor:
        chunk_ctx, chunk_valid = self.select_local_chunk(
            sender_ctx=sender_ctx,
            sender_valid=sender_valid,
            anchor_patch_start=anchor_patch_start,
            sender_patch_start=sender_patch_start,
        )
        q = self.build_completion_query(masked_anchor_local, anchor_patch_hms, chunk_ctx, chunk_valid)
        k = self.key_mlp(chunk_ctx)
        v = self.value_mlp(chunk_ctx)
        attn_scores = (q.unsqueeze(2) * k).sum(dim=-1) / (self.cfg.hid_dim ** 0.5)
        no_candidate = ~chunk_valid.any(dim=-1)
        neg_large = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~chunk_valid, neg_large)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        completed_tokens = (attn_weights.unsqueeze(-1) * v).sum(dim=2)
        completed_tokens = self.completion_norm(completed_tokens + q)
        gate = self.completion_gate(torch.cat([masked_anchor_local, completed_tokens, q], dim=-1))
        completed_tokens = gate * completed_tokens + (1.0 - gate) * masked_anchor_local
        completed_tokens = torch.where(no_candidate.unsqueeze(-1), masked_anchor_local, completed_tokens)
        mask_f = masked_patch.unsqueeze(-1).float()
        return masked_anchor_local * (1.0 - mask_f) + completed_tokens * mask_f

    def extract_last_obs(self, anchor_patch: torch.Tensor, anchor_mask: torch.Tensor) -> torch.Tensor:
        bsz = anchor_patch.size(0)
        flat_patch = anchor_patch.view(bsz, -1, self.cfg.input_dim)
        flat_mask = anchor_mask.view(bsz, -1)
        out = []
        for b in range(bsz):
            valid_idx = torch.where(flat_mask[b] > 0)[0]
            if valid_idx.numel() > 0:
                out.append(flat_patch[b, valid_idx[-1]])
            else:
                out.append(flat_patch[b, -1])
        return torch.stack(out, dim=0)

    def build_future_queries(
        self,
        last_obs: torch.Tensor,
        anchor_ctx: torch.Tensor,
        anchor_valid: torch.Tensor,
        future_dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        counts = torch.clamp(anchor_valid.sum(dim=1).long(), min=1, max=self.cfg.npatch)
        last_patch = counts - 1
        batch_idx = torch.arange(last_obs.size(0), device=last_obs.device)
        last_ctx = anchor_ctx[batch_idx, last_patch]
        global_anchor = self.global_proj(torch.cat([last_obs, last_ctx], dim=-1))
        denom = math.log1p(max(float(self.cfg.max_future_dt_for_embed), 1.0))
        dt_feat = torch.log1p(future_dt.clamp(min=0.0)).div(denom).unsqueeze(-1)
        dt_embed = self.future_time_embed(dt_feat)
        step_ids = torch.arange(self.cfg.future_steps, device=last_obs.device)
        step_embed = self.future_step_embed(step_ids).unsqueeze(0).expand(last_obs.size(0), -1, -1)
        global_expand = global_anchor.unsqueeze(1).expand(-1, self.cfg.future_steps, -1)
        return self.future_query_proj(torch.cat([global_expand, dt_embed, step_embed], dim=-1)), dt_embed

    def decode_motion(
        self,
        last_obs: torch.Tensor,
        anchor_ctx: torch.Tensor,
        anchor_valid: torch.Tensor,
        future_dt: torch.Tensor,
    ) -> torch.Tensor:
        queries, dt_embed = self.build_future_queries(last_obs, anchor_ctx, anchor_valid, future_dt)
        attended, _ = self.future_cross_attn(
            query=queries,
            key=anchor_ctx,
            value=anchor_ctx,
            key_padding_mask=anchor_valid <= 0,
            need_weights=False,
        )
        attended = self.future_decoder_norm(attended + queries)
        last_obs_expand = last_obs.unsqueeze(1).expand(-1, self.cfg.future_steps, -1)
        decoder_in = torch.cat([attended, queries, last_obs_expand, dt_embed], dim=-1)
        return self.step_decoder(decoder_in)

    def forward(self, return_loss: bool = False, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            _, original_anchor_valid = self.encode_stream(batch["anchor_patch"], batch["anchor_mask"], batch["anchor_patch_hms"])
            masked_patch = self.sample_masked_patches(original_anchor_valid)
        masked_anchor_patch, masked_anchor_mask = self.apply_training_corruption(
            batch["anchor_patch"],
            batch["anchor_mask"],
            masked_patch,
        )

        target_anchor_local, _ = self.encode_stream(batch["anchor_patch"], batch["anchor_mask"], batch["anchor_patch_hms"])
        masked_anchor_local, observed_anchor_valid = self.encode_stream(masked_anchor_patch, masked_anchor_mask, batch["anchor_patch_hms"])
        sender_local, sender_valid = self.encode_stream(batch["sender_patch"], batch["sender_mask"], batch["sender_patch_hms"])
        sender_ctx = self.contextualize(sender_local, sender_valid, self.sender_transformer, self.sender_patch_attn)
        completed_anchor = self.complete_anchor(
            masked_anchor_local=masked_anchor_local,
            sender_ctx=sender_ctx,
            sender_valid=sender_valid,
            masked_patch=masked_patch,
            anchor_patch_hms=batch["anchor_patch_hms"],
            anchor_patch_start=batch["anchor_patch_start"],
            sender_patch_start=batch["sender_patch_start"],
        )
        completed_valid = torch.maximum(observed_anchor_valid, masked_patch.float())
        anchor_ctx = self.contextualize(completed_anchor, completed_valid, self.anchor_transformer, self.anchor_patch_attn)
        last_obs = self.extract_last_obs(masked_anchor_patch, masked_anchor_mask)
        pred_motion_norm = self.decode_motion(last_obs, anchor_ctx, completed_valid, batch["future_dt"])

        out = {
            "pred_motion_norm": pred_motion_norm,
            "masked_patch": masked_patch,
            "target_anchor_local": target_anchor_local,
            "completed_anchor": completed_anchor,
        }
        if not return_loss:
            return out

        pred_motion_physical = pred_motion_norm * self.motion_scale_buffer.view(1, 1, -1)
        losses = compute_motion_losses(
            pred_motion_norm=pred_motion_norm,
            gt_motion_norm=batch["future_motion_norm"],
            pred_motion_physical=pred_motion_physical,
            gt_future_pos_norm=batch["future_pos_norm"],
            gt_future_pos=batch["future_pos"],
            future_dt=batch["future_dt"],
            position_scale=batch["position_scale"],
            target_mode=self.cfg.target_mode,
            lambda_motion=self.cfg.lambda_motion,
            lambda_traj=self.cfg.lambda_traj,
            lambda_final=self.cfg.lambda_final,
            loss_name=self.cfg.motion_loss,
        )
        completion_loss = pred_motion_norm.sum() * 0.0
        if masked_patch.any():
            completion_loss = F.mse_loss(completed_anchor[masked_patch], target_anchor_local[masked_patch], reduction="mean")
        total = losses["loss"] + self.cfg.lambda_completion * completion_loss
        return {
            **losses,
            "loss": total,
            "completion_loss": completion_loss,
            "pred_motion_norm": pred_motion_norm,
            "pred_motion_physical": pred_motion_physical,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pair patch cross-attention completion model.")
    parser.add_argument("--input-jsonl", type=str, default="data/raw/dual_ais_level2_shape_aggressiveC_pointclean_ema_recomputed_outlierdrop200.jsonl")
    parser.add_argument("--workdir", type=str, default="workdir_pair_cross_completion")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Optional existing cache directory containing train/val/test.npz and dataset_meta.json.",
    )
    parser.add_argument("--exp-name", type=str, default="pair_patch_cross_completion")
    parser.add_argument("--anchor-key", type=str, default="traj_a")
    parser.add_argument("--sender-key", type=str, default="traj_b")
    parser.add_argument("--npatch", type=int, default=8)
    parser.add_argument("--patch-minutes", type=int, default=15)
    parser.add_argument("--max-points-per-patch", type=int, default=16)
    parser.add_argument("--future-steps", type=int, default=12)
    parser.add_argument("--window-stride-points", type=int, default=4)
    parser.add_argument(
        "--no-sliding-window",
        action="store_true",
        help="Use one sample per original trajectory pair by predicting the final future segment.",
    )
    parser.add_argument("--min-anchor-points", type=int, default=12)
    parser.add_argument("--min-sender-points", type=int, default=8)
    parser.add_argument("--target-mode", choices=["velocity", "displacement"], default="velocity")
    parser.add_argument("--scale-stat", choices=["p95", "std"], default="p95")
    parser.add_argument("--min-future-dt", type=float, default=5.0)
    parser.add_argument("--max-abs-velocity", type=float, default=200.0)
    parser.add_argument("--max-abs-displacement", type=float, default=10000.0)
    parser.add_argument("--max-raw-pairs", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hid-dim", type=int, default=128)
    parser.add_argument("--te-dim", type=int, default=16)
    parser.add_argument("--future-te-dim", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--tf-layer", type=int, default=2)
    parser.add_argument("--tau-seconds", type=float, default=300.0)
    parser.add_argument("--decoder-dropout", type=float, default=0.1)
    parser.add_argument("--lambda-motion", type=float, default=1.0)
    parser.add_argument("--lambda-traj", type=float, default=1.0)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--lambda-completion", type=float, default=1.0)
    parser.add_argument("--mask-patch-count", type=int, default=1)
    parser.add_argument("--local-chunk-size", type=int, default=3)
    parser.add_argument("--sparse-training-prob", type=float, default=0.5)
    parser.add_argument("--sparse-keep-points", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision on CUDA.")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP even on CUDA.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cuDNN on Ampere+ GPUs.")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_runtime(args: argparse.Namespace, device: torch.device) -> Dict[str, bool]:
    use_tf32 = bool(args.tf32 or (device.type == "cuda" and not args.no_tf32))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    if args.amp:
        use_amp = device.type == "cuda"
    return {"use_amp": use_amp, "use_tf32": use_tf32}


def move_batch_fast(batch: Dict[str, torch.Tensor], device: torch.device, position_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    out = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}
    out["position_scale"] = position_scale
    return out


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    motion_scale: torch.Tensor,
    position_scale: torch.Tensor,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Dict[str, float]:
    model.train(optimizer is not None)
    totals = {
        "loss": 0.0,
        "motion_loss": 0.0,
        "traj_loss": 0.0,
        "final_loss": 0.0,
        "completion_loss": 0.0,
        "pos_mae": 0.0,
        "final_mae": 0.0,
        "n": 0,
    }
    for batch in loader:
        batch = move_batch_fast(batch, device, position_scale)
        model.set_motion_scale(motion_scale)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            out = model(return_loss=True, **batch)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        bsz = batch["anchor_patch"].size(0)
        totals["n"] += bsz
        for key in ("loss", "motion_loss", "traj_loss", "final_loss", "completion_loss", "pos_mae", "final_mae"):
            totals[key] += float(out[key].detach().item()) * bsz
    denom = max(totals["n"], 1)
    return {k: (v / denom if k != "n" else v) for k, v in totals.items()}


def train_model(
    model: nn.Module,
    exp_dir: Path,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    motion_scale: torch.Tensor,
    position_scale: torch.Tensor,
    lr: float,
    weight_decay: float,
    run_config: Dict,
    use_amp: bool,
) -> Dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log_path = exp_dir / "train_log.jsonl"
    stdout_path = exp_dir / "stdout.jsonl"
    best_path = exp_dir / "best.pt"
    last_path = exp_dir / "last.pt"
    (exp_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    best_val = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, motion_scale, position_scale, use_amp, scaler)
        val_metrics = run_epoch(model, val_loader, None, device, motion_scale, position_scale, use_amp, None)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        append_log(log_path, record)
        append_log(stdout_path, record)
        print(json.dumps(record), flush=True)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": run_config}
        torch.save(state, last_path)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(state, best_path)

    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    test_metrics = run_epoch(model, test_loader, None, device, motion_scale, position_scale, use_amp, None)
    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "log_file": str(log_path),
        "history": history,
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    append_log(stdout_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    runtime_cfg = configure_cuda_runtime(args, device)
    log_device_info(device)
    print(json.dumps({"runtime": runtime_cfg}), flush=True)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (workdir / "data_cache")
    print(json.dumps({"status": "build_or_load_cache_start", "cache_dir": str(cache_dir)}), flush=True)
    meta = build_or_load_cache(args, cache_dir)
    print(json.dumps({"status": "build_or_load_cache_done", "splits": meta.get("splits", {})}), flush=True)

    exp_dir = build_experiment_dir(workdir, args.exp_name)
    print(json.dumps({"status": "experiment_dir_created", "experiment_dir": str(exp_dir)}), flush=True)
    train_set = PairPatchDataset(str(cache_dir / "train.npz"))
    val_set = PairPatchDataset(str(cache_dir / "val.npz"))
    test_set = PairPatchDataset(str(cache_dir / "test.npz"))
    print(
        json.dumps(
            {
                "status": "datasets_loaded",
                "train_size": len(train_set),
                "val_size": len(val_set),
                "test_size": len(test_set),
            }
        ),
        flush=True,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    cfg = CrossCompletionConfig(
        input_dim=6,
        npatch=args.npatch,
        max_points_per_patch=args.max_points_per_patch,
        future_steps=args.future_steps,
        target_mode=args.target_mode,
        hid_dim=args.hid_dim,
        te_dim=args.te_dim,
        future_te_dim=args.future_te_dim,
        nhead=args.nhead,
        tf_layer=args.tf_layer,
        patch_minutes=args.patch_minutes,
        tau_seconds=args.tau_seconds,
        decoder_dropout=args.decoder_dropout,
        lambda_motion=args.lambda_motion,
        lambda_traj=args.lambda_traj,
        lambda_final=args.lambda_final,
        lambda_completion=args.lambda_completion,
        mask_patch_count=args.mask_patch_count,
        local_chunk_size=args.local_chunk_size,
        sparse_training_prob=args.sparse_training_prob,
        sparse_keep_points=args.sparse_keep_points,
    )
    model = PairPatchCrossCompletionModel(cfg).to(device)
    motion_scale = torch.tensor(meta["motion_scale"], dtype=torch.float32, device=device)
    position_scale = torch.tensor(meta["position_scale"], dtype=torch.float32, device=device)

    run_config = {
        "args": vars(args),
        "dataset_meta": meta,
        "model_config": asdict(cfg),
        "experiment_dir": str(exp_dir),
        "training_note": "Randomly mask full anchor patches; build a time-conditioned completion query from masked patch hms, patch index, and local sender-chunk summaries; then cross-attend over a strict local sender chunk of three nearest valid patches.",
    }
    summary = train_model(
        model=model,
        exp_dir=exp_dir,
        epochs=args.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        motion_scale=motion_scale,
        position_scale=position_scale,
        lr=args.lr,
        weight_decay=args.weight_decay,
        run_config=run_config,
        use_amp=runtime_cfg["use_amp"],
    )
    (exp_dir / "root_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        failure_log = os.environ.get("PAIR_PATCH_FAILURE_LOG", "")
        payload = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps({"fatal": payload}, ensure_ascii=True), flush=True)
        if failure_log:
            failure_path = Path(failure_log)
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        raise
