import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_dataset import MotionDataset
from models import MotionPatchTimeQueryForecaster, TimeQueryMotionPatchConfig
from trajectory_eval_utils import evaluate_trajectory_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Traj-PathFormer on single-trajectory no-missing data.")
    parser.add_argument("--data-dir", type=str, default="data/singletraj_pair")
    parser.add_argument("--workdir", type=str, default="workdir_ours_singletraj_noslide")
    parser.add_argument("--exp-name", type=str, default="ours_timequery_noslide")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hid-dim", type=int, default=128)
    parser.add_argument("--te-dim", type=int, default=16)
    parser.add_argument("--future-te-dim", type=int, default=16)
    parser.add_argument("--nlayer", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--tf-layer", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lambda-motion", type=float, default=1.0)
    parser.add_argument("--lambda-traj", type=float, default=1.0)
    parser.add_argument("--lambda-final", type=float, default=1.0)
    parser.add_argument("--motion-loss", type=str, default="smoothl1", choices=["smoothl1", "mse"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(items: List[Dict]) -> Dict[str, torch.Tensor]:
    keys = [
        "history",
        "history_raw",
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
    moved = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
    moved["position_scale"] = position_scale
    return moved


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    lr: float,
    min_lr: float,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(int(steps_per_epoch * epochs), 1)
    warmup_steps = max(int(steps_per_epoch * warmup_epochs), 1)
    floor = max(float(min_lr) / max(float(lr), 1e-12), 0.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(float(step + 1) / float(warmup_steps), floor)
        progress = min(max(float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1)), 0.0), 1.0)
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    model: MotionPatchTimeQueryForecaster,
    loader: DataLoader,
    motion_scale: torch.Tensor,
    position_scale: torch.Tensor,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
    grad_clip: float = 1.0,
    max_batches: int = 0,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "motion_loss": 0.0,
        "traj_loss": 0.0,
        "final_loss": 0.0,
        "pos_mae": 0.0,
        "final_mae": 0.0,
    }
    n_items = 0
    n_batches = 0

    for raw_batch in loader:
        n_batches += 1
        if max_batches > 0 and n_batches > max_batches:
            break
        batch = move_batch(raw_batch, device, position_scale)
        bsz = int(batch["history"].size(0))

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            losses = model.compute_loss(batch, motion_scale=motion_scale)
            loss = losses["loss"]

        if is_train:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        n_items += bsz
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * bsz

    out = {key: val / max(n_items, 1) for key, val in totals.items()}
    out["n"] = n_items
    out["batches"] = max(n_batches - (1 if max_batches > 0 and n_batches > max_batches else 0), 0)
    if optimizer is not None:
        out["lr"] = float(optimizer.param_groups[0]["lr"])
    return out


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir).resolve()
    datasets = {split: MotionDataset(str(data_dir / f"{split}.npz")) for split in ("train", "val", "test")}
    meta = json.loads((data_dir / "dataset_meta.json").read_text())

    input_dim = int(datasets["train"].history.shape[-1])
    future_steps = int(datasets["train"].future_pos.shape[1])
    history_steps = int(datasets["train"].history.shape[1])
    npatch = 8
    patch_len = history_steps // npatch
    if npatch * patch_len != history_steps:
        raise ValueError(f"history length {history_steps} is not divisible by npatch={npatch}.")

    cfg = TimeQueryMotionPatchConfig(
        input_dim=input_dim,
        future_steps=future_steps,
        target_mode=str(datasets["train"].target_mode),
        motion_dim=2,
        npatch=npatch,
        patch_len=patch_len,
        hid_dim=int(args.hid_dim),
        te_dim=int(args.te_dim),
        nlayer=int(args.nlayer),
        nhead=int(args.nhead),
        tf_layer=int(args.tf_layer),
        decoder_dropout=float(args.dropout),
        motion_loss=str(args.motion_loss),
        lambda_motion=float(args.lambda_motion),
        lambda_traj=float(args.lambda_traj),
        lambda_final=float(args.lambda_final),
        future_te_dim=int(args.future_te_dim),
        use_kinematic_prior=True,
    )

    model = MotionPatchTimeQueryForecaster(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    loaders = {
        split: DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=(split == "train"),
            num_workers=int(args.num_workers),
            collate_fn=collate,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        for split, ds in datasets.items()
    }
    steps_per_epoch = max(len(loaders["train"]), 1)
    scheduler = make_scheduler(
        optimizer,
        lr=float(args.lr),
        min_lr=float(args.min_lr),
        steps_per_epoch=steps_per_epoch,
        epochs=int(args.epochs),
        warmup_epochs=float(args.warmup_epochs),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    use_amp = bool(args.amp and device.type == "cuda")

    motion_scale = torch.tensor(meta["motion_scale"], dtype=torch.float32, device=device)
    position_scale = torch.tensor(meta["position_scale"], dtype=torch.float32, device=device)
    model.set_motion_scale(motion_scale)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(args.workdir).resolve() / "experiments" / f"{args.exp_name}_{stamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "args": vars(args),
        "dataset_meta": meta,
        "model_type": "MotionPatchTimeQueryForecaster",
        "model_config": cfg.__dict__,
        "experiment_dir": str(exp_dir),
    }
    save_json(exp_dir / "config.json", config)
    log_path = exp_dir / "train_log.jsonl"

    best_val = float("inf")
    best_epoch = -1
    history = []
    stale_epochs = 0
    for epoch in range(int(args.epochs)):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            motion_scale,
            position_scale,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=float(args.grad_clip),
            max_batches=int(args.max_train_batches),
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                loaders["val"],
                motion_scale,
                position_scale,
                device,
                use_amp=False,
                max_batches=int(args.max_eval_batches),
            )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics}, exp_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = int(epoch)
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_metrics}, exp_dir / "best.pt")
        else:
            stale_epochs += 1
            if int(args.patience) > 0 and stale_epochs >= int(args.patience):
                break

    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            loaders["test"],
            motion_scale,
            position_scale,
            device,
            use_amp=False,
            max_batches=int(args.max_eval_batches),
        )
    latlon_metrics = evaluate_trajectory_metrics(model, datasets["test"], motion_scale, position_scale, device)
    summary = {
        "model": "Ours",
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "test_latlon_metrics": latlon_metrics,
        "checkpoint_best": str((exp_dir / "best.pt").resolve()),
        "checkpoint_last": str((exp_dir / "last.pt").resolve()),
        "log_file": str(log_path.resolve()),
        "history": history,
    }
    save_json(exp_dir / "summary.json", summary)
    print(json.dumps({"summary": str((exp_dir / "summary.json").resolve()), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
