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
from models.patch_forecaster import compute_motion_losses


EARTH_METERS_PER_LON = 111320.0
EARTH_METERS_PER_LAT = 110540.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Table 2 completion protocols on pair patch data.")
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-counts", type=str, default="1,2,3")
    parser.add_argument("--sparse-keep-points", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--include-last-patch", action="store_true")
    parser.add_argument("--out-dir", type=str, default="")
    return parser.parse_args()


def local_xy_to_latlon(x: float, y: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    lat = lat_ref + (y / EARTH_METERS_PER_LAT)
    lon = lon_ref + (x / (EARTH_METERS_PER_LON * math.cos(math.radians(lat_ref))))
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


def load_model(exp_dir: Path, device: torch.device) -> Tuple[PairPatchCrossCompletionModel, Dict]:
    config = json.loads((exp_dir / "config.json").read_text())
    cfg = CrossCompletionConfig(**config["model_config"])
    model = PairPatchCrossCompletionModel(cfg).to(device)
    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def truncate_dataset(dataset: PairPatchDataset, max_samples: int) -> None:
    if max_samples <= 0 or max_samples >= len(dataset):
        return
    for name in (
        "sample_ids",
        "anchor_patch",
        "anchor_patch_raw",
        "anchor_mask",
        "anchor_patch_hms",
        "anchor_patch_start",
        "sender_patch",
        "sender_patch_raw",
        "sender_mask",
        "sender_patch_hms",
        "sender_patch_start",
        "future_dt",
        "future_pos",
        "future_pos_norm",
        "future_motion_norm",
        "anchor_latlon",
    ):
        setattr(dataset, name, getattr(dataset, name)[:max_samples])


def choose_patch_indices(valid_patch_mask: np.ndarray, count: int, rng: np.random.Generator, include_last: bool) -> np.ndarray:
    valid_idx = np.where(valid_patch_mask > 0)[0]
    if not include_last and valid_idx.size > 0:
        valid_idx = valid_idx[valid_idx < (valid_patch_mask.shape[0] - 1)]
    if valid_idx.size == 0:
        return np.zeros((0,), dtype=np.int64)
    count = min(int(count), int(valid_idx.size))
    return np.sort(rng.choice(valid_idx, size=count, replace=False).astype(np.int64))


def build_mask_plan(dataset: PairPatchDataset, patch_counts: List[int], seed: int, include_last: bool) -> Dict[int, List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    plan = {}
    for count in patch_counts:
        per_item = []
        for idx in range(len(dataset)):
            valid_patch = (dataset.anchor_mask[idx].sum(axis=-1) > 0).astype(np.float32)
            per_item.append(choose_patch_indices(valid_patch, count, rng, include_last))
        plan[count] = per_item
    return plan


def collate(dataset: PairPatchDataset, indices: List[int], device: torch.device, position_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
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
    return batch


def build_forced_mask(mask_indices: List[np.ndarray], npatch: int, device: torch.device) -> torch.Tensor:
    forced = torch.zeros((len(mask_indices), npatch), dtype=torch.bool, device=device)
    for b, patch_idx in enumerate(mask_indices):
        if len(patch_idx) > 0:
            forced[b, torch.as_tensor(patch_idx, dtype=torch.long, device=device)] = True
    return forced


def apply_sparse_patch_mask(
    anchor_patch: torch.Tensor,
    anchor_mask: torch.Tensor,
    forced_mask: torch.Tensor,
    keep_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sparse_mask = anchor_mask.clone()
    keep_points = max(int(keep_points), 1)
    for b in range(anchor_mask.size(0)):
        patch_ids = torch.where(forced_mask[b])[0]
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


def corrupt_anchor(
    anchor_patch: torch.Tensor,
    anchor_mask: torch.Tensor,
    forced_mask: torch.Tensor,
    corruption: str,
    sparse_keep_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if corruption == "empty":
        keep_patch = (1.0 - forced_mask.float()).unsqueeze(-1).unsqueeze(-1)
        keep_point = (1.0 - forced_mask.float()).unsqueeze(-1)
        return anchor_patch * keep_patch, anchor_mask * keep_point
    if corruption == "sparse":
        return apply_sparse_patch_mask(anchor_patch, anchor_mask, forced_mask, sparse_keep_points)
    raise ValueError(f"Unsupported corruption: {corruption}")


def direct_source_fill(
    corrupted_patch: torch.Tensor,
    corrupted_mask: torch.Tensor,
    sender_patch: torch.Tensor,
    sender_mask: torch.Tensor,
    anchor_patch_start: torch.Tensor,
    sender_patch_start: torch.Tensor,
    forced_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    filled_patch = corrupted_patch.clone()
    filled_mask = corrupted_mask.clone()
    sender_valid = sender_mask.sum(dim=-1) > 0
    dist = (anchor_patch_start.unsqueeze(-1) - sender_patch_start.unsqueeze(1)).abs()
    dist = dist.masked_fill(~sender_valid.unsqueeze(1), float("inf"))
    nearest = dist.argmin(dim=-1)
    for b in range(corrupted_patch.size(0)):
        patch_ids = torch.where(forced_mask[b])[0]
        for p in patch_ids.tolist():
            q = int(nearest[b, p].item())
            if sender_valid[b, q]:
                filled_patch[b, p] = sender_patch[b, q]
                filled_mask[b, p] = sender_mask[b, q]
    return filled_patch, filled_mask


def forecast_with_protocol(
    model: PairPatchCrossCompletionModel,
    batch: Dict[str, torch.Tensor],
    forced_mask: torch.Tensor,
    corruption: str,
    protocol: str,
    motion_scale: torch.Tensor,
    sparse_keep_points: int,
) -> Dict[str, torch.Tensor]:
    model.set_motion_scale(motion_scale)
    corrupted_patch, corrupted_mask = corrupt_anchor(
        batch["anchor_patch"], batch["anchor_mask"], forced_mask, corruption, sparse_keep_points
    )

    if protocol == "direct_source_fill":
        anchor_patch, anchor_mask = direct_source_fill(
            corrupted_patch,
            corrupted_mask,
            batch["sender_patch"],
            batch["sender_mask"],
            batch["anchor_patch_start"],
            batch["sender_patch_start"],
            forced_mask,
        )
        anchor_local, anchor_valid = model.encode_stream(anchor_patch, anchor_mask, batch["anchor_patch_hms"])
        anchor_ctx = model.contextualize(anchor_local, anchor_valid, model.anchor_transformer, model.anchor_patch_attn)
        last_obs = model.extract_last_obs(anchor_patch, anchor_mask)
    elif protocol == "single_source_cross_attn":
        masked_anchor_local, observed_anchor_valid = model.encode_stream(corrupted_patch, corrupted_mask, batch["anchor_patch_hms"])
        sender_local, sender_valid = model.encode_stream(batch["sender_patch"], batch["sender_mask"], batch["sender_patch_hms"])
        sender_ctx = model.contextualize(sender_local, sender_valid, model.sender_transformer, model.sender_patch_attn)
        completed_anchor = model.complete_anchor(
            masked_anchor_local=masked_anchor_local,
            sender_ctx=sender_ctx,
            sender_valid=sender_valid,
            masked_patch=forced_mask,
            anchor_patch_hms=batch["anchor_patch_hms"],
            anchor_patch_start=batch["anchor_patch_start"],
            sender_patch_start=batch["sender_patch_start"],
        )
        completed_valid = torch.maximum(observed_anchor_valid, forced_mask.float())
        anchor_ctx = model.contextualize(completed_anchor, completed_valid, model.anchor_transformer, model.anchor_patch_attn)
        last_obs = model.extract_last_obs(corrupted_patch, corrupted_mask)
        anchor_valid = completed_valid
    elif protocol == "none":
        anchor_local, anchor_valid = model.encode_stream(corrupted_patch, corrupted_mask, batch["anchor_patch_hms"])
        anchor_ctx = model.contextualize(anchor_local, anchor_valid, model.anchor_transformer, model.anchor_patch_attn)
        last_obs = model.extract_last_obs(corrupted_patch, corrupted_mask)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    pred_motion_norm = model.decode_motion(last_obs, anchor_ctx, anchor_valid, batch["future_dt"])
    pred_motion_physical = pred_motion_norm * motion_scale.view(1, 1, -1)
    return compute_motion_losses(
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


def evaluate_setting(
    model: PairPatchCrossCompletionModel,
    dataset: PairPatchDataset,
    mask_plan: List[np.ndarray],
    corruption: str,
    protocol: str,
    motion_scale: torch.Tensor,
    position_scale: torch.Tensor,
    batch_size: int,
    device: torch.device,
    sparse_keep_points: int,
) -> Dict[str, float]:
    totals = {"loss": 0.0, "pos_mae": 0.0, "final_mae": 0.0}
    l2_values = []
    mse_values = []
    dtw_values = []
    n_items = 0

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            indices = list(range(start, end))
            batch = collate(dataset, indices, device, position_scale)
            forced_mask = build_forced_mask(mask_plan[start:end], model.cfg.npatch, device)
            out = forecast_with_protocol(
                model=model,
                batch=batch,
                forced_mask=forced_mask,
                corruption=corruption,
                protocol=protocol,
                motion_scale=motion_scale,
                sparse_keep_points=sparse_keep_points,
            )
            bsz = len(indices)
            n_items += bsz
            totals["loss"] += float(out["loss"].detach().cpu()) * bsz
            totals["pos_mae"] += float(out["pos_mae"].detach().cpu()) * bsz
            totals["final_mae"] += float(out["final_mae"].detach().cpu()) * bsz

            pred_xy = out["pred_future_pos"].detach().cpu().numpy().astype(np.float64)
            gt_xy = dataset.future_pos[indices].astype(np.float64)
            anchor_latlon = dataset.anchor_latlon[indices].astype(np.float64)
            for pred, gt, latlon in zip(pred_xy, gt_xy, anchor_latlon):
                pred_latlon = trajectory_local_xy_to_latlon(pred, latlon)
                gt_latlon = trajectory_local_xy_to_latlon(gt, latlon)
                diff = pred_latlon - gt_latlon
                l2_values.append(float(np.linalg.norm(diff, axis=-1).mean()))
                mse_values.append(float(np.mean(diff ** 2)))
                dtw_values.append(float(dtw_distance(pred_latlon, gt_latlon)))

    return {
        "test_loss": totals["loss"] / max(n_items, 1),
        "pos_mae": totals["pos_mae"] / max(n_items, 1),
        "final_mae": totals["final_mae"] / max(n_items, 1),
        "avg_l2": float(np.mean(l2_values)) if l2_values else 0.0,
        "avg_mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "avg_dtw": float(np.mean(dtw_values)) if dtw_values else 0.0,
        "n": int(n_items),
    }


def write_csv(rows: List[Dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corruption",
                "patches",
                "protocol",
                "test_loss",
                "pos_mae",
                "final_mae",
                "avg_l2",
                "avg_mse",
                "avg_dtw",
                "n",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows: List[Dict], path: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Quick dual-AIS robustness to empty and sparse historical patches. Lower is better for all metrics.}",
        "    \\label{tab:missing-sparse-robustness-quick}",
        "    \\small",
        "    \\begin{tabular}{llp{0.24\\textwidth}cccccc}",
        "        \\hline",
        "        Corruption & Patches & Completion Protocol & Test Loss & Pos MAE & Final MAE & Avg L2 & Avg MSE & Avg DTW \\\\",
        "        \\hline",
    ]
    for row in rows:
        lines.append(
            "        {corruption} & {patches} & {protocol} & {test_loss:.4f} & {pos_mae:.4f} & "
            "{final_mae:.4f} & {avg_l2:.6f} & {avg_mse:.8f} & {avg_dtw:.6f} \\\\".format(**row)
        )
    lines.extend(["        \\hline", "    \\end{tabular}", "\\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    exp_dir = Path(args.exp_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    patch_counts = [int(x.strip()) for x in args.patch_counts.split(",") if x.strip()]
    protocols = [
        ("none", "None"),
        ("direct_source_fill", "Direct Source Fill"),
        ("single_source_cross_attn", "Single-Source Cross-Attn."),
    ]

    model, config = load_model(exp_dir, device)
    dataset = PairPatchDataset(str(cache_dir / f"{args.split}.npz"))
    truncate_dataset(dataset, int(args.max_samples))
    motion_scale = torch.tensor(config["dataset_meta"]["motion_scale"], dtype=torch.float32, device=device)
    position_scale = torch.tensor(config["dataset_meta"]["position_scale"], dtype=torch.float32, device=device)
    mask_plan = build_mask_plan(dataset, patch_counts, int(args.seed), bool(args.include_last_patch))

    rows = []
    results = {
        "source": "dual_ais_rawpair_no_sliding",
        "split": args.split,
        "coordinate_space": "latlon_for_l2_mse_dtw",
        "patch_counts": patch_counts,
        "sparse_keep_points": int(args.sparse_keep_points),
        "include_last_patch": bool(args.include_last_patch),
        "rows": [],
    }

    for corruption in ("empty", "sparse"):
        for count in patch_counts:
            for protocol_key, protocol_label in protocols:
                metrics = evaluate_setting(
                    model=model,
                    dataset=dataset,
                    mask_plan=mask_plan[count],
                    corruption=corruption,
                    protocol=protocol_key,
                    motion_scale=motion_scale,
                    position_scale=position_scale,
                    batch_size=int(args.batch_size),
                    device=device,
                    sparse_keep_points=int(args.sparse_keep_points),
                )
                row = {
                    "corruption": corruption.capitalize(),
                    "patches": count,
                    "protocol": protocol_label,
                    **metrics,
                }
                rows.append(row)
                results["rows"].append(row)
                print(json.dumps(row), flush=True)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (exp_dir / "table2_protocols")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "table2_protocol_results.json"
    csv_path = out_dir / "table2_protocol_results.csv"
    tex_path = out_dir / "table2_protocol_results.tex"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "tex": str(tex_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
