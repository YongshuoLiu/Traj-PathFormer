import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from pair_patch_data import PairPatchDataset
from train_pair_patch_cross_completion import CrossCompletionConfig, PairPatchCrossCompletionModel


EARTH_METERS_PER_LON = 111320.0
EARTH_METERS_PER_LAT = 110540.0


def local_xy_to_latlon(x: float, y: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    lat = lat_ref + (y / EARTH_METERS_PER_LAT)
    lat_ref_rad = math.radians(lat_ref)
    lon = lon_ref + (x / (EARTH_METERS_PER_LON * math.cos(lat_ref_rad)))
    return float(lat), float(lon)


def trajectory_local_xy_to_latlon(xy: np.ndarray, anchor_latlon: np.ndarray) -> np.ndarray:
    out = np.zeros_like(xy, dtype=np.float64)
    for i, (x, y) in enumerate(xy):
        out[i] = local_xy_to_latlon(float(x), float(y), float(anchor_latlon[0]), float(anchor_latlon[1]))
    return out


def dtw_distance(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    n_pred, n_gt = pred_xy.shape[0], gt_xy.shape[0]
    dp = np.full((n_pred + 1, n_gt + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n_pred + 1):
        for j in range(1, n_gt + 1):
            cost = float(np.linalg.norm(pred_xy[i - 1] - gt_xy[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n_pred, n_gt])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mask robustness for pair patch cross-completion model.")
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args()


def load_model(exp_dir: Path, device: torch.device) -> Tuple[PairPatchCrossCompletionModel, Dict]:
    config = json.loads((exp_dir / "config.json").read_text())
    model_cfg = CrossCompletionConfig(**config["model_config"])
    model = PairPatchCrossCompletionModel(model_cfg).to(device)
    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def maybe_truncate_dataset(dataset: PairPatchDataset, max_samples: int) -> None:
    if not max_samples or max_samples <= 0:
        return
    dataset.sample_ids = dataset.sample_ids[:max_samples]
    dataset.anchor_patch = dataset.anchor_patch[:max_samples]
    dataset.anchor_patch_raw = dataset.anchor_patch_raw[:max_samples]
    dataset.anchor_mask = dataset.anchor_mask[:max_samples]
    dataset.anchor_patch_hms = dataset.anchor_patch_hms[:max_samples]
    dataset.anchor_patch_start = dataset.anchor_patch_start[:max_samples]
    dataset.sender_patch = dataset.sender_patch[:max_samples]
    dataset.sender_patch_raw = dataset.sender_patch_raw[:max_samples]
    dataset.sender_mask = dataset.sender_mask[:max_samples]
    dataset.sender_patch_hms = dataset.sender_patch_hms[:max_samples]
    dataset.sender_patch_start = dataset.sender_patch_start[:max_samples]
    dataset.future_dt = dataset.future_dt[:max_samples]
    dataset.future_pos = dataset.future_pos[:max_samples]
    dataset.future_pos_norm = dataset.future_pos_norm[:max_samples]
    dataset.future_motion_norm = dataset.future_motion_norm[:max_samples]
    dataset.anchor_latlon = dataset.anchor_latlon[:max_samples]


def choose_patch_indices(valid_patch_mask: np.ndarray, mask_patch_count: int, rng: np.random.Generator) -> np.ndarray:
    valid_idx = np.where(valid_patch_mask > 0)[0]
    if valid_idx.size == 0:
        return np.zeros((0,), dtype=np.int64)
    count = min(mask_patch_count, int(valid_idx.size))
    chosen = rng.choice(valid_idx, size=count, replace=False)
    return np.sort(chosen.astype(np.int64))


def build_mask_plan(dataset: PairPatchDataset, mask_patch_counts: List[int], seed: int) -> Dict[int, List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    plan = {}
    for count in mask_patch_counts:
        per_dataset = []
        for idx in range(len(dataset)):
            patch_valid = (dataset.anchor_mask[idx].sum(axis=-1) > 0).astype(np.float32)
            per_dataset.append(choose_patch_indices(patch_valid, count, rng))
        plan[count] = per_dataset
    return plan


def collate_batch(
    dataset: PairPatchDataset,
    indices: List[int],
    mask_patch_indices_list: List[np.ndarray],
    device: torch.device,
    position_scale: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], List[np.ndarray]]:
    batch = {
        "anchor_patch": torch.from_numpy(dataset.anchor_patch[indices]).to(device),
        "anchor_mask": torch.from_numpy(dataset.anchor_mask[indices]).to(device),
        "anchor_patch_hms": torch.from_numpy(dataset.anchor_patch_hms[indices]).to(device),
        "anchor_patch_start": torch.from_numpy(dataset.anchor_patch_start[indices]).to(device),
        "sender_patch": torch.from_numpy(dataset.sender_patch[indices]).to(device),
        "sender_mask": torch.from_numpy(dataset.sender_mask[indices]).to(device),
        "sender_patch_hms": torch.from_numpy(dataset.sender_patch_hms[indices]).to(device),
        "sender_patch_start": torch.from_numpy(dataset.sender_patch_start[indices]).to(device),
        "future_dt": torch.from_numpy(dataset.future_dt[indices]).to(device),
        "future_pos": torch.from_numpy(dataset.future_pos[indices]).to(device),
        "future_pos_norm": torch.from_numpy(dataset.future_pos_norm[indices]).to(device),
        "future_motion_norm": torch.from_numpy(dataset.future_motion_norm[indices]).to(device),
        "position_scale": position_scale,
    }
    return batch, mask_patch_indices_list


def forced_forward(
    model: PairPatchCrossCompletionModel,
    batch: Dict[str, torch.Tensor],
    forced_masked_patch: torch.Tensor,
    motion_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    model.set_motion_scale(motion_scale)
    masked_anchor_patch, masked_anchor_mask = model.apply_patch_mask(
        batch["anchor_patch"],
        batch["anchor_mask"],
        forced_masked_patch,
    )
    target_anchor_local, _ = model.encode_stream(batch["anchor_patch"], batch["anchor_mask"], batch["anchor_patch_hms"])
    masked_anchor_local, observed_anchor_valid = model.encode_stream(
        masked_anchor_patch,
        masked_anchor_mask,
        batch["anchor_patch_hms"],
    )
    sender_local, sender_valid = model.encode_stream(batch["sender_patch"], batch["sender_mask"], batch["sender_patch_hms"])
    sender_ctx = model.contextualize(sender_local, sender_valid, model.sender_transformer, model.sender_patch_attn)
    completed_anchor = model.complete_anchor(
        masked_anchor_local=masked_anchor_local,
        sender_ctx=sender_ctx,
        sender_valid=sender_valid,
        masked_patch=forced_masked_patch,
        anchor_patch_hms=batch["anchor_patch_hms"],
        anchor_patch_start=batch["anchor_patch_start"],
        sender_patch_start=batch["sender_patch_start"],
    )
    completed_valid = torch.maximum(observed_anchor_valid, forced_masked_patch.float())
    anchor_ctx = model.contextualize(completed_anchor, completed_valid, model.anchor_transformer, model.anchor_patch_attn)
    last_obs = model.extract_last_obs(masked_anchor_patch, masked_anchor_mask)
    pred_motion_norm = model.decode_motion(last_obs, anchor_ctx, completed_valid, batch["future_dt"])
    pred_motion_physical = pred_motion_norm * motion_scale.view(1, 1, -1)
    from models.patch_forecaster import compute_motion_losses

    losses = compute_motion_losses(
        pred_motion_norm=pred_motion_norm,
        gt_motion_norm=batch["future_motion_norm"],
        pred_motion_physical=pred_motion_physical,
        gt_future_pos_norm=batch["future_pos_norm"],
        gt_future_pos=batch["future_pos"],
        future_dt=batch["future_dt"],
        position_scale=batch["position_scale"],
        target_mode=model.cfg.target_mode,
        lambda_motion=model.cfg.lambda_motion,
        lambda_traj=model.cfg.lambda_traj,
        lambda_final=model.cfg.lambda_final,
        loss_name=model.cfg.motion_loss,
    )
    return losses


def evaluate_for_mask_count(
    model: PairPatchCrossCompletionModel,
    dataset: PairPatchDataset,
    mask_plan_for_count: List[np.ndarray],
    motion_scale: torch.Tensor,
    position_scale: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    l2_values = []
    mse_values = []
    dtw_values = []

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            indices = list(range(start, end))
            batch_mask_indices = mask_plan_for_count[start:end]
            batch, _ = collate_batch(dataset, indices, batch_mask_indices, device, position_scale)
            forced_mask = torch.zeros((len(indices), model.cfg.npatch), dtype=torch.bool, device=device)
            for b, patch_idx in enumerate(batch_mask_indices):
                if len(patch_idx) > 0:
                    forced_mask[b, torch.as_tensor(patch_idx, dtype=torch.long, device=device)] = True
            out = forced_forward(model, batch, forced_mask, motion_scale)
            pred_xy = out["pred_future_pos"].detach().cpu().numpy().astype(np.float64)
            gt_xy = dataset.future_pos[indices].astype(np.float64)
            anchor_latlon = dataset.anchor_latlon[indices].astype(np.float64)

            for pred, gt, latlon in zip(pred_xy, gt_xy, anchor_latlon):
                pred_latlon = trajectory_local_xy_to_latlon(pred, latlon)
                gt_latlon = trajectory_local_xy_to_latlon(gt, latlon)
                l2_values.append(float(np.linalg.norm(pred_latlon - gt_latlon, axis=-1).mean()))
                mse_values.append(float(np.mean((pred_latlon - gt_latlon) ** 2)))
                dtw_values.append(float(dtw_distance(pred_latlon, gt_latlon)))

    return {
        "avg_l2": float(np.mean(l2_values)) if l2_values else 0.0,
        "avg_mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "avg_dtw": float(np.mean(dtw_values)) if dtw_values else 0.0,
    }


def write_csv(rows: List[Dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["masked_patch_count", "metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows: List[Dict], path: Path) -> None:
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("    \\centering")
    lines.append("    \\caption{Mask robustness evaluation for the pair patch cross-completion model.}")
    lines.append("    \\label{tab:pair-patch-cross-mask-robustness}")
    lines.append("    \\begin{tabular}{lcc}")
    lines.append("        \\toprule")
    lines.append("        Masked Patches / Metric & Value \\\\")
    lines.append("        \\midrule")
    for row in rows:
        label = f"{row['masked_patch_count']} patch {row['metric'].upper()}"
        lines.append(f"        {label} & {row['value']:.6f} \\\\")
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    exp_dir = Path(args.exp_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    model, config = load_model(exp_dir, device)
    dataset = PairPatchDataset(str(cache_dir / f"{args.split}.npz"))
    maybe_truncate_dataset(dataset, int(args.max_samples))
    motion_scale = torch.tensor(config["dataset_meta"]["motion_scale"], dtype=torch.float32, device=device)
    position_scale = torch.tensor(config["dataset_meta"]["position_scale"], dtype=torch.float32, device=device)

    mask_patch_counts = list(range(1, int(config["model_config"]["npatch"])))
    mask_plan = build_mask_plan(dataset, mask_patch_counts, int(args.seed))
    results = {"coordinate_space": "latlon", "mask_mode": "random_full_patch_multi_query_cross_attention", "mask_results": {}}
    rows = []

    for count in mask_patch_counts:
        metrics = evaluate_for_mask_count(
            model=model,
            dataset=dataset,
            mask_plan_for_count=mask_plan[count],
            motion_scale=motion_scale,
            position_scale=position_scale,
            batch_size=int(args.batch_size),
            device=device,
        )
        key = f"{count:02d}_patch"
        results["mask_results"][key] = {
            "masked_patch_count": count,
            "metrics": metrics,
        }
        for metric_name in ("avg_l2", "avg_mse", "avg_dtw"):
            rows.append(
                {
                    "masked_patch_count": count,
                    "metric": metric_name.replace("avg_", ""),
                    "value": metrics[metric_name],
                }
            )

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (exp_dir / "mask_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pair_patch_cross_mask_robustness.json"
    csv_path = out_dir / "pair_patch_cross_mask_robustness.csv"
    tex_path = out_dir / "pair_patch_cross_mask_robustness.tex"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "tex": str(tex_path)}, indent=2))


if __name__ == "__main__":
    main()
