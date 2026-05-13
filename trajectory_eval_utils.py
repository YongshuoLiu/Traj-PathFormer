import math
from typing import Dict, Tuple

import numpy as np
import torch

EARTH_METERS_PER_LON = 111320.0
EARTH_METERS_PER_LAT = 110540.0


def build_single_item_batch(item: dict, device: torch.device, position_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    batch = {
        "history": torch.from_numpy(item["history"]).unsqueeze(0).to(device),
        "history_mask": torch.from_numpy(item["history_mask"]).unsqueeze(0).to(device),
        "future_dt": torch.from_numpy(item["future_dt"]).unsqueeze(0).to(device),
        "future_pos": torch.from_numpy(item["future_pos"]).unsqueeze(0).to(device),
        "future_pos_norm": torch.from_numpy(item["future_pos_norm"]).unsqueeze(0).to(device),
        "future_motion_norm": torch.from_numpy(item["future_motion"]).unsqueeze(0).to(device),
        "position_scale": position_scale,
    }
    if "history_raw" in item:
        batch["history_raw"] = torch.from_numpy(item["history_raw"]).unsqueeze(0).to(device)
    return batch


def dtw_distance(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    n_pred, n_gt = pred_xy.shape[0], gt_xy.shape[0]
    dp = np.full((n_pred + 1, n_gt + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n_pred + 1):
        for j in range(1, n_gt + 1):
            cost = float(np.linalg.norm(pred_xy[i - 1] - gt_xy[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n_pred, n_gt])


def local_xy_to_latlon(x: float, y: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    lat = lat_ref + (y / EARTH_METERS_PER_LAT)
    lat_ref_rad = math.radians(lat_ref)
    lon = lon_ref + (x / (EARTH_METERS_PER_LON * math.cos(lat_ref_rad)))
    return float(lat), float(lon)


def trajectory_local_xy_to_latlon(xy: np.ndarray, anchor_latlon: np.ndarray) -> np.ndarray:
    lat_ref = float(anchor_latlon[0])
    lon_ref = float(anchor_latlon[1])
    out = np.zeros_like(xy, dtype=np.float64)
    for i, (x, y) in enumerate(xy):
        lat, lon = local_xy_to_latlon(float(x), float(y), lat_ref, lon_ref)
        out[i, 0] = lat
        out[i, 1] = lon
    return out


def evaluate_trajectory_metrics(model, dataset, motion_scale: torch.Tensor, position_scale: torch.Tensor, device: torch.device) -> Dict[str, float]:
    pos_mae_values = []
    final_mae_values = []
    pos_mse_values = []
    final_mse_values = []
    l2_values = []
    mse_values = []
    dtw_values = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            batch = build_single_item_batch(item, device, position_scale)
            out = model.compute_loss(batch, motion_scale=motion_scale)

            pred_xy = out["pred_future_pos"][0].detach().cpu().numpy().astype(np.float64)
            gt_xy = item["future_pos"].astype(np.float64)
            anchor_latlon = item["anchor_latlon"].astype(np.float64)

            pred_latlon = trajectory_local_xy_to_latlon(pred_xy, anchor_latlon)
            gt_latlon = trajectory_local_xy_to_latlon(gt_xy, anchor_latlon)
            diff = pred_latlon - gt_latlon

            pos_mae_values.append(float(np.abs(diff).mean()))
            final_mae_values.append(float(np.abs(diff[-1]).mean()))
            pos_mse_values.append(float(np.mean(diff ** 2)))
            final_mse_values.append(float(np.mean(diff[-1] ** 2)))
            l2_values.append(float(np.linalg.norm(diff, axis=-1).mean()))
            mse_values.append(float(np.mean(diff ** 2)))
            dtw_values.append(float(dtw_distance(pred_latlon, gt_latlon)))

    return {
        "coordinate_space": "latlon",
        "pos_mae": float(np.mean(pos_mae_values)) if pos_mae_values else 0.0,
        "final_mae": float(np.mean(final_mae_values)) if final_mae_values else 0.0,
        "pos_mse": float(np.mean(pos_mse_values)) if pos_mse_values else 0.0,
        "final_mse": float(np.mean(final_mse_values)) if final_mse_values else 0.0,
        "avg_l2": float(np.mean(l2_values)) if l2_values else 0.0,
        "avg_mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "avg_dtw": float(np.mean(dtw_values)) if dtw_values else 0.0,
        "num_samples": int(len(dataset)),
    }
