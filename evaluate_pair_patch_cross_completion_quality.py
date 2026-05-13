import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pair_patch_data import PairPatchDataset
from train_pair_patch_cross_completion import CrossCompletionConfig, PairPatchCrossCompletionModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate completion quality for pair patch cross-completion model.")
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
    return np.sort(rng.choice(valid_idx, size=count, replace=False).astype(np.int64))


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


def evaluate_completion_for_mask_count(
    model: PairPatchCrossCompletionModel,
    dataset: PairPatchDataset,
    mask_plan_for_count: List[np.ndarray],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    mse_values = []
    mae_values = []
    cosine_values = []
    token_count = 0

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            indices = list(range(start, end))
            batch_mask_indices = mask_plan_for_count[start:end]

            anchor_patch = torch.from_numpy(dataset.anchor_patch[indices]).to(device)
            anchor_mask = torch.from_numpy(dataset.anchor_mask[indices]).to(device)
            anchor_patch_hms = torch.from_numpy(dataset.anchor_patch_hms[indices]).to(device)
            anchor_patch_start = torch.from_numpy(dataset.anchor_patch_start[indices]).to(device)
            sender_patch = torch.from_numpy(dataset.sender_patch[indices]).to(device)
            sender_mask = torch.from_numpy(dataset.sender_mask[indices]).to(device)
            sender_patch_hms = torch.from_numpy(dataset.sender_patch_hms[indices]).to(device)
            sender_patch_start = torch.from_numpy(dataset.sender_patch_start[indices]).to(device)

            forced_mask = torch.zeros((len(indices), model.cfg.npatch), dtype=torch.bool, device=device)
            for b, patch_idx in enumerate(batch_mask_indices):
                if len(patch_idx) > 0:
                    forced_mask[b, torch.as_tensor(patch_idx, dtype=torch.long, device=device)] = True

            masked_anchor_patch, masked_anchor_mask = model.apply_patch_mask(anchor_patch, anchor_mask, forced_mask)
            target_anchor_local, _ = model.encode_stream(anchor_patch, anchor_mask, anchor_patch_hms)
            masked_anchor_local, _ = model.encode_stream(masked_anchor_patch, masked_anchor_mask, anchor_patch_hms)
            sender_local, sender_valid = model.encode_stream(sender_patch, sender_mask, sender_patch_hms)
            sender_ctx = model.contextualize(sender_local, sender_valid, model.sender_transformer, model.sender_patch_attn)
            completed_anchor = model.complete_anchor(
                masked_anchor_local=masked_anchor_local,
                sender_ctx=sender_ctx,
                sender_valid=sender_valid,
                masked_patch=forced_mask,
                anchor_patch_hms=anchor_patch_hms,
                anchor_patch_start=anchor_patch_start,
                sender_patch_start=sender_patch_start,
            )

            if forced_mask.any():
                pred_sel = completed_anchor[forced_mask]
                tgt_sel = target_anchor_local[forced_mask]
                mse_values.append(float(F.mse_loss(pred_sel, tgt_sel, reduction="mean").item()))
                mae_values.append(float(F.l1_loss(pred_sel, tgt_sel, reduction="mean").item()))
                cosine_values.append(float(F.cosine_similarity(pred_sel, tgt_sel, dim=-1).mean().item()))
                token_count += int(pred_sel.shape[0])

    return {
        "completion_mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "completion_mae": float(np.mean(mae_values)) if mae_values else 0.0,
        "completion_cosine": float(np.mean(cosine_values)) if cosine_values else 0.0,
        "masked_token_count": token_count,
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
    lines.append("    \\caption{Completion-quality evaluation for the pair patch cross-completion model.}")
    lines.append("    \\label{tab:pair-patch-cross-completion-quality}")
    lines.append("    \\begin{tabular}{lcc}")
    lines.append("        \\toprule")
    lines.append("        Masked Patches / Metric & Value \\\\")
    lines.append("        \\midrule")
    for row in rows:
        label = f"{row['masked_patch_count']} patch {row['metric']}"
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

    mask_patch_counts = list(range(1, int(config["model_config"]["npatch"])))
    mask_plan = build_mask_plan(dataset, mask_patch_counts, int(args.seed))
    results = {"mask_mode": "random_full_patch_localwindow_completion", "completion_results": {}}
    rows = []

    for count in mask_patch_counts:
        metrics = evaluate_completion_for_mask_count(
            model=model,
            dataset=dataset,
            mask_plan_for_count=mask_plan[count],
            batch_size=int(args.batch_size),
            device=device,
        )
        key = f"{count:02d}_patch"
        results["completion_results"][key] = {"masked_patch_count": count, "metrics": metrics}
        for metric_name in ("completion_mse", "completion_mae", "completion_cosine"):
            rows.append(
                {
                    "masked_patch_count": count,
                    "metric": metric_name,
                    "value": metrics[metric_name],
                }
            )

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (exp_dir / "completion_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "pair_patch_cross_completion_quality.json"
    csv_path = out_dir / "pair_patch_cross_completion_quality.csv"
    tex_path = out_dir / "pair_patch_cross_completion_quality.tex"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "tex": str(tex_path)}, indent=2))


if __name__ == "__main__":
    main()
