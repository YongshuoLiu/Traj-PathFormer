import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_dataset import MotionDataset
from models.baseline_forecasters import BaselineMotionConfig, build_baseline_model
from trajectory_eval_utils import evaluate_trajectory_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick single-trajectory baseline suite.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--workdir", type=str, default="workdir_quick_singletraj_noslide")
    parser.add_argument("--models", type=str, default="agdn,tlstm,tpatch")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hid-dim", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collate(items: List[Dict]) -> Dict[str, torch.Tensor]:
    keys = [
        "history",
        "history_mask",
        "future_dt",
        "future_pos",
        "future_pos_norm",
        "future_motion",
    ]
    batch = {}
    for key in keys:
        batch[key] = torch.as_tensor(np.stack([item[key] for item in items], axis=0), dtype=torch.float32)
    batch["future_motion_norm"] = batch["future_motion"]
    return batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device, position_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    out = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
    out["position_scale"] = position_scale
    return out


def run_epoch(
    model,
    loader: DataLoader,
    motion_scale: torch.Tensor,
    position_scale: torch.Tensor,
    device: torch.device,
    optimizer=None,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {"loss": 0.0, "motion_loss": 0.0, "traj_loss": 0.0, "final_loss": 0.0, "pos_mae": 0.0, "final_mae": 0.0}
    n = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device, position_scale)
        if train:
            optimizer.zero_grad(set_to_none=True)
        losses = model.compute_loss(batch, motion_scale=motion_scale)
        loss = losses["loss"]
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bsz = int(batch["history"].size(0))
        n += bsz
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * bsz
    out = {key: (val / max(n, 1)) for key, val in totals.items()}
    out["n"] = n
    return out


def train_one(model_name: str, args: argparse.Namespace, datasets: Dict[str, MotionDataset], meta: Dict, device: torch.device) -> Dict:
    model_key = model_name.lower().strip()
    model_key = "tpatch" if model_key in {"t-patchgnn", "tpatchgnn", "tpatch"} else model_key
    cfg = BaselineMotionConfig(
        model_name=model_key,
        input_dim=int(datasets["train"].history.shape[-1]),
        future_steps=int(datasets["train"].future_pos.shape[1]),
        target_mode=str(datasets["train"].target_mode),
        npatch=8,
        patch_len=int(datasets["train"].history.shape[1] // 8),
        hid_dim=int(args.hid_dim),
        nlayer=2,
        nhead=4,
        tf_layer=1,
    )
    model = build_baseline_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)

    loaders = {
        split: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
        )
        for split, ds in datasets.items()
    }
    motion_scale = torch.tensor(meta["motion_scale"], dtype=torch.float32, device=device)
    position_scale = torch.tensor(meta["position_scale"], dtype=torch.float32, device=device)

    exp_dir = Path(args.workdir) / "experiments" / f"{model_key}_e{args.epochs}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "model_config": cfg.__dict__, "dataset_meta": meta}, indent=2)
    )

    best_val = float("inf")
    best_epoch = -1
    history = []
    for epoch in range(int(args.epochs)):
        train_metrics = run_epoch(model, loaders["train"], motion_scale, position_scale, device, optimizer=optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, loaders["val"], motion_scale, position_scale, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps({"model": model_key, **row}), flush=True)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics}, exp_dir / "best.pt")

    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    with torch.no_grad():
        test_metrics = run_epoch(model, loaders["test"], motion_scale, position_scale, device)
    latlon_metrics = evaluate_trajectory_metrics(model, datasets["test"], motion_scale, position_scale, device)
    summary = {
        "baseline_model": model_key,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "test_latlon_metrics": latlon_metrics,
        "checkpoint_best": str((exp_dir / "best.pt").resolve()),
        "history": history,
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    datasets = {split: MotionDataset(str(data_dir / f"{split}.npz")) for split in ("train", "val", "test")}
    meta = json.loads((data_dir / "dataset_meta.json").read_text())
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name in [x.strip() for x in args.models.split(",") if x.strip()]:
        try:
            results[model_name] = train_one(model_name, args, datasets, meta, device)
        except Exception as exc:
            results[model_name] = {"error": str(exc)}
            print(json.dumps({"model": model_name, "error": str(exc)}), flush=True)

    output = Path(args.workdir) / "quick_suite_summary.json"
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps({"summary": str(output.resolve()), "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
