import argparse
import json
from pathlib import Path

import torch

from motion_dataset import MotionDataset
from models import (
    MotionPatchConfig,
    MotionPatchPretrainForecaster,
    MotionPatchTimeQueryForecaster,
    TimeQueryMotionPatchConfig,
)
from trajectory_eval_utils import evaluate_trajectory_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate average L2, MSE, and DTW on a trajectory forecast split.")
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory containing config.json and checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Checkpoint file name inside exp-dir.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=str, default="", help="Optional explicit path for metrics json output.")
    return parser.parse_args()


def load_model(exp_dir: Path, checkpoint_name: str, device: torch.device):
    config = json.loads((exp_dir / "config.json").read_text())
    model_type = config.get("model_type", "MotionPatchPretrainForecaster")
    if model_type == "MotionPatchTimeQueryForecaster":
        model_cfg = TimeQueryMotionPatchConfig(**config["model_config"])
        model = MotionPatchTimeQueryForecaster(model_cfg).to(device)
    else:
        model_cfg = MotionPatchConfig(**config["model_config"])
        model = MotionPatchPretrainForecaster(model_cfg).to(device)
    ckpt = torch.load(exp_dir / checkpoint_name, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    device = torch.device(args.device)

    model, config = load_model(exp_dir, args.checkpoint, device)
    data_dir = Path(config["args"]["data_dir"])
    dataset = MotionDataset(str(data_dir / f"{args.split}.npz"))

    motion_scale = torch.tensor(config["dataset_meta"]["motion_scale"], dtype=torch.float32, device=device)
    position_scale = torch.tensor(config["dataset_meta"]["position_scale"], dtype=torch.float32, device=device)

    metrics = {
        "experiment_dir": str(exp_dir),
        "checkpoint": str((exp_dir / args.checkpoint).resolve()),
        "split": args.split,
        **evaluate_trajectory_metrics(model, dataset, motion_scale, position_scale, device),
    }

    if args.output_json:
        output_json = Path(args.output_json).resolve()
    else:
        output_json = exp_dir / "visualizations" / f"{args.split}_metrics.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
