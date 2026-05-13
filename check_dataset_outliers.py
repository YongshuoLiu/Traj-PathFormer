import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check outlier points in preprocessed trajectory datasets.")
    parser.add_argument("--data-dir", type=str, default="data/singletraj_pair")
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=None,
        help="Outlier threshold in m/s. If omitted, read max_abs_velocity from dataset_meta.json.",
    )
    parser.add_argument("--topk", type=int, default=10, help="Show top-K samples with the most outlier points.")
    return parser.parse_args()


def history_outlier_stats(blob, velocity_threshold: float, topk: int) -> Dict:
    history_raw = blob["history_raw"]
    history_mask = blob["history_mask"] > 0
    sample_ids = blob["sample_ids"]

    delta = history_raw[:, 1:, :2] - history_raw[:, :-1, :2]
    dt = history_raw[:, 1:, 5]
    valid = history_mask[:, 1:] & history_mask[:, :-1] & (dt > 0)
    speed = np.linalg.norm(delta, axis=-1) / np.maximum(dt, EPS)
    outlier = valid & (speed > velocity_threshold)

    per_sample_counts = outlier.sum(axis=1)
    top_indices = np.argsort(-per_sample_counts)[: min(len(per_sample_counts), max(topk, 0))]
    top_examples: List[Dict] = []
    for idx in top_indices:
        if per_sample_counts[idx] <= 0:
            break
        sample_speed = np.where(valid[idx], speed[idx], np.nan)
        top_examples.append(
            {
                "sample_index": int(idx),
                "sample_id": str(sample_ids[idx]),
                "outlier_point_count": int(per_sample_counts[idx]),
                "max_speed_mps": float(np.nanmax(sample_speed)),
            }
        )

    return {
        "samples_with_outlier": int((per_sample_counts > 0).sum()),
        "total_samples": int(history_raw.shape[0]),
        "sample_ratio": float((per_sample_counts > 0).mean()),
        "outlier_points": int(outlier.sum()),
        "total_candidate_points": int(valid.sum()),
        "point_ratio": float(outlier.sum() / max(valid.sum(), 1)),
        "top_examples": top_examples,
    }


def future_outlier_stats(blob, velocity_threshold: float) -> Dict:
    future_pos = blob["future_pos"]
    future_dt = blob["future_dt"]

    prev = np.concatenate([np.zeros((future_pos.shape[0], 1, 2), dtype=future_pos.dtype), future_pos[:, :-1, :]], axis=1)
    delta = future_pos - prev
    valid = future_dt > 0
    speed = np.linalg.norm(delta, axis=-1) / np.maximum(future_dt, EPS)
    outlier = valid & (speed > velocity_threshold)
    per_sample_counts = outlier.sum(axis=1)

    return {
        "samples_with_outlier": int((per_sample_counts > 0).sum()),
        "total_samples": int(future_pos.shape[0]),
        "sample_ratio": float((per_sample_counts > 0).mean()),
        "outlier_points": int(outlier.sum()),
        "total_candidate_points": int(valid.sum()),
        "point_ratio": float(outlier.sum() / max(valid.sum(), 1)),
    }


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "dataset_meta.json").read_text())
    velocity_threshold = float(args.velocity_threshold or meta["max_abs_velocity"])

    report = {
        "data_dir": str(data_dir.resolve()),
        "velocity_threshold_mps": velocity_threshold,
        "criterion": "A point is flagged as an outlier if the incoming step speed exceeds the threshold.",
        "splits": {},
    }

    total_samples = 0
    total_samples_with_outlier = 0
    total_points = 0
    total_outlier_points = 0

    for split in ("train", "val", "test"):
        blob = np.load(data_dir / f"{split}.npz", allow_pickle=False)
        history_stats = history_outlier_stats(blob, velocity_threshold, args.topk)
        future_stats = future_outlier_stats(blob, velocity_threshold)

        overall_sample_mask_count = max(history_stats["samples_with_outlier"], future_stats["samples_with_outlier"])
        # Exact overall sample count: union over per-split sample flags.
        history_raw = blob["history_raw"]
        history_mask = blob["history_mask"] > 0
        delta_h = history_raw[:, 1:, :2] - history_raw[:, :-1, :2]
        dt_h = history_raw[:, 1:, 5]
        valid_h = history_mask[:, 1:] & history_mask[:, :-1] & (dt_h > 0)
        out_h = valid_h & (np.linalg.norm(delta_h, axis=-1) / np.maximum(dt_h, EPS) > velocity_threshold)

        future_pos = blob["future_pos"]
        future_dt = blob["future_dt"]
        prev_f = np.concatenate([np.zeros((future_pos.shape[0], 1, 2), dtype=future_pos.dtype), future_pos[:, :-1, :]], axis=1)
        out_f = (future_dt > 0) & (
            np.linalg.norm(future_pos - prev_f, axis=-1) / np.maximum(future_dt, EPS) > velocity_threshold
        )
        sample_has_outlier = out_h.any(axis=1) | out_f.any(axis=1)

        overall_stats = {
            "samples_with_outlier": int(sample_has_outlier.sum()),
            "total_samples": int(sample_has_outlier.shape[0]),
            "sample_ratio": float(sample_has_outlier.mean()),
            "outlier_points": int(out_h.sum() + out_f.sum()),
            "total_candidate_points": int(valid_h.sum() + (future_dt > 0).sum()),
            "point_ratio": float((out_h.sum() + out_f.sum()) / max(int(valid_h.sum() + (future_dt > 0).sum()), 1)),
        }

        report["splits"][split] = {
            "history": history_stats,
            "future": future_stats,
            "overall": overall_stats,
        }

        total_samples += overall_stats["total_samples"]
        total_samples_with_outlier += overall_stats["samples_with_outlier"]
        total_points += overall_stats["total_candidate_points"]
        total_outlier_points += overall_stats["outlier_points"]

    report["overall"] = {
        "samples_with_outlier": total_samples_with_outlier,
        "total_samples": total_samples,
        "sample_ratio": total_samples_with_outlier / max(total_samples, 1),
        "outlier_points": total_outlier_points,
        "total_candidate_points": total_points,
        "point_ratio": total_outlier_points / max(total_points, 1),
    }

    out_path = data_dir / f"outlier_report_v{int(round(velocity_threshold))}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
