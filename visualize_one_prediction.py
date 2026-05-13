import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from motion_dataset import MotionDataset
from models import MotionPatchConfig, MotionPatchPretrainForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one trajectory forecast from a trained patch model.")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory containing config.json and best.pt.")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint file name inside exp-dir.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index to visualize.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_model(exp_dir: Path, checkpoint_name: str, device: torch.device):
    config = json.loads((exp_dir / "config.json").read_text())
    model_cfg = MotionPatchConfig(**config["model_config"])
    model = MotionPatchPretrainForecaster(model_cfg).to(device)
    ckpt = torch.load(exp_dir / checkpoint_name, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def prepare_batch(item: dict, device: torch.device, position_scale: torch.Tensor) -> dict:
    return {
        "history": torch.from_numpy(item["history"]).unsqueeze(0).to(device),
        "history_mask": torch.from_numpy(item["history_mask"]).unsqueeze(0).to(device),
        "future_dt": torch.from_numpy(item["future_dt"]).unsqueeze(0).to(device),
        "future_pos": torch.from_numpy(item["future_pos"]).unsqueeze(0).to(device),
        "future_pos_norm": torch.from_numpy(item["future_pos_norm"]).unsqueeze(0).to(device),
        "future_motion_norm": torch.from_numpy(item["future_motion"]).unsqueeze(0).to(device),
        "position_scale": position_scale,
    }


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    device = torch.device(args.device)

    model, config = load_model(exp_dir, args.checkpoint, device)
    data_dir = Path(config["args"]["data_dir"])
    dataset = MotionDataset(str(data_dir / f"{args.split}.npz"))
    item = dataset[args.sample_index]

    motion_scale = torch.tensor(config["dataset_meta"]["motion_scale"], dtype=torch.float32, device=device)
    position_scale = torch.tensor(config["dataset_meta"]["position_scale"], dtype=torch.float32, device=device)
    batch = prepare_batch(item, device, position_scale)

    with torch.no_grad():
        out = model.compute_loss(batch, motion_scale=motion_scale)

    history_mask = item["history_mask"] > 0
    history_xy = item["history_raw"][history_mask, :2]
    gt_xy = item["future_pos"]
    pred_xy = out["pred_future_pos"][0].cpu().numpy()

    pos_mae = float(np.abs(pred_xy - gt_xy).mean())
    final_mae = float(np.abs(pred_xy[-1] - gt_xy[-1]).mean())

    out_dir = exp_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_id = str(item["sample_id"])
    stem = f"{args.split}_idx{args.sample_index:05d}"
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(history_xy[:, 0], history_xy[:, 1], color="#4C78A8", linewidth=2.0, marker="o", markersize=2.5, label="History")
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#59A14F", linewidth=2.0, marker="o", markersize=3.0, label="GT Future")
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#E15759", linewidth=2.0, marker="x", markersize=4.0, label="Pred Future")
    ax.scatter([0.0], [0.0], color="black", s=36, label="Anchor / Last Obs", zorder=5)
    ax.set_title(f"{sample_id}\n{args.split} idx={args.sample_index} | pos_mae={pos_mae:.2f} m | final_mae={final_mae:.2f} m")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    summary = {
        "experiment_dir": str(exp_dir),
        "checkpoint": str((exp_dir / args.checkpoint).resolve()),
        "split": args.split,
        "sample_index": int(args.sample_index),
        "sample_id": sample_id,
        "anchor_latlon": np.asarray(item["anchor_latlon"]).tolist(),
        "pos_mae": pos_mae,
        "final_mae": final_mae,
        "png_path": str(png_path),
        "history_xy_tail": history_xy[-5:].tolist(),
        "gt_xy": gt_xy.tolist(),
        "pred_xy": pred_xy.tolist(),
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
