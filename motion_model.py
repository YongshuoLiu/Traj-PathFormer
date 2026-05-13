from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUMotionForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        future_steps: int,
        motion_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.future_steps = int(future_steps)
        self.motion_dim = int(motion_dim)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, future_steps * motion_dim),
        )

    def forward(self, history: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(history)
        h = h * history_mask.unsqueeze(-1)

        lengths = history_mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        last_hidden = hidden[-1]
        out = self.head(last_hidden)
        return out.view(history.size(0), self.future_steps, self.motion_dim)


def reconstruct_positions(
    pred_motion_physical: torch.Tensor,
    future_dt: torch.Tensor,
    target_mode: str,
) -> torch.Tensor:
    if target_mode == "velocity":
        step_delta = pred_motion_physical * future_dt.unsqueeze(-1)
    elif target_mode == "displacement":
        step_delta = pred_motion_physical
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    return torch.cumsum(step_delta, dim=1)


def compute_losses(
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
    loss_name: str = "smoothl1",
) -> Dict[str, torch.Tensor]:
    if loss_name == "mse":
        point_loss = F.mse_loss
    elif loss_name == "smoothl1":
        point_loss = F.smooth_l1_loss
    else:
        raise ValueError(f"Unsupported loss_name: {loss_name}")

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
        "motion_loss": motion_loss.detach(),
        "traj_loss": traj_loss.detach(),
        "final_loss": final_loss.detach(),
        "pos_mae": pos_mae.detach(),
        "final_mae": final_mae.detach(),
    }
