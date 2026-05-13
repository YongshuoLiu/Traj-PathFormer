import argparse
import json
import math
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


EARTH_METERS_PER_LON = 111320.0
EARTH_METERS_PER_LAT = 110540.0
EPS = 1e-6
HISTORY_CLIP_VALUE = 5.0
HISTORY_X_IDX = 0
HISTORY_Y_IDX = 1
HISTORY_SPEED_IDX = 2
HISTORY_COURSE_SIN_IDX = 3
HISTORY_COURSE_COS_IDX = 4
HISTORY_DT_IDX = 5
HISTORY_SOURCE_IDX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess AIS jsonl into motion-target datasets.")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="data/raw/dual_ais_level2_shape_aggressiveC_pointclean_ema_recomputed.jsonl",
    )
    parser.add_argument("--output-dir", type=str, default="data/singletraj_pair")
    parser.add_argument(
        "--source-mode",
        choices=["pair", "traj_a", "traj_b"],
        default="pair",
        help="Which trajectory stream to expose to the model. "
             "'pair' merges both AIS streams; 'traj_a'/'traj_b' keeps only one stream.",
    )
    parser.add_argument("--history-steps", type=int, default=64)
    parser.add_argument("--future-steps", type=int, default=12)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--target-mode", choices=["velocity", "displacement"], default="velocity")
    parser.add_argument("--scale-stat", choices=["p95", "std"], default="p95")
    parser.add_argument("--min-future-dt", type=float, default=5.0)
    parser.add_argument("--max-abs-velocity", type=float, default=200.0)
    parser.add_argument("--max-abs-displacement", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-train-items",
        type=int,
        default=0,
        help="If > 0, cap the train split to this many windows by subsampling the full window pool before splitting.",
    )
    return parser.parse_args()


def latlon_to_local_xy(lat: float, lon: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    lat_ref_rad = math.radians(lat_ref)
    x = (lon - lon_ref) * EARTH_METERS_PER_LON * math.cos(lat_ref_rad)
    y = (lat - lat_ref) * EARTH_METERS_PER_LAT
    return x, y


def merge_events(obj: Dict, source_mode: str = "pair") -> List[Dict]:
    order = obj["features"]["order"]
    time_idx = order.index("time")
    lat_idx = order.index("lat")
    lon_idx = order.index("lon")
    vel_idx = order.index("vel")
    cou_idx = order.index("cou")

    events = []
    source_mode = str(source_mode).lower()
    if source_mode == "pair":
        source_specs = (("traj_a", 0.0), ("traj_b", 1.0))
    elif source_mode == "traj_a":
        source_specs = (("traj_a", 0.0),)
    elif source_mode == "traj_b":
        source_specs = (("traj_b", 1.0),)
    else:
        raise ValueError(f"Unsupported source_mode: {source_mode}")

    for source_name, source_value in source_specs:
        for row in obj["features"].get(source_name, []):
            events.append(
                {
                    "t": float(row[time_idx]),
                    "lat": float(row[lat_idx]),
                    "lon": float(row[lon_idx]),
                    "speed": float(row[vel_idx]),
                    "course": float(row[cou_idx]),
                    "source": float(source_value),
                }
            )
    events.sort(key=lambda x: (x["t"], x["source"]))
    return events


def build_window_features(
    events: Sequence[Dict],
    history_steps: int,
    future_steps: int,
    target_mode: str,
    min_future_dt: float,
    max_abs_velocity: float,
    max_abs_displacement: float,
) -> Dict:
    if len(events) < history_steps + future_steps:
        return {}

    history_events = list(events[:history_steps])
    future_events = list(events[history_steps : history_steps + future_steps])
    anchor = history_events[-1]
    lat_ref = anchor["lat"]
    lon_ref = anchor["lon"]

    all_events = history_events + future_events
    positions = []
    for event in all_events:
        x, y = latlon_to_local_xy(event["lat"], event["lon"], lat_ref, lon_ref)
        positions.append((x, y))

    history_pos = positions[:history_steps]
    future_pos = positions[history_steps:]
    history = []
    prev_t = history_events[0]["t"]
    for idx, event in enumerate(history_events):
        x, y = history_pos[idx]
        dt = 0.0 if idx == 0 else max(event["t"] - prev_t, 0.0)
        prev_t = event["t"]
        course_rad = math.radians(event["course"])
        history.append(
            [
                x,
                y,
                event["speed"],
                math.sin(course_rad),
                math.cos(course_rad),
                dt,
                event["source"],
            ]
        )

    future_dt = []
    future_motion = []
    prev_t = history_events[-1]["t"]
    prev_x, prev_y = 0.0, 0.0
    for idx, event in enumerate(future_events):
        cur_x, cur_y = future_pos[idx]
        dt = max(event["t"] - prev_t, 0.0)
        dx = cur_x - prev_x
        dy = cur_y - prev_y
        if target_mode == "velocity":
            if dt < min_future_dt:
                return {}
            inv_dt = 1.0 / max(dt, EPS)
            motion = [dx * inv_dt, dy * inv_dt]
            if max(abs(motion[0]), abs(motion[1])) > max_abs_velocity:
                return {}
        else:
            motion = [dx, dy]
            if max(abs(motion[0]), abs(motion[1])) > max_abs_displacement:
                return {}
        future_dt.append(dt)
        future_motion.append(motion)
        prev_t = event["t"]
        prev_x, prev_y = cur_x, cur_y

    future_motion_arr = np.asarray(future_motion, dtype=np.float32)
    if not np.isfinite(future_motion_arr).all():
        return {}

    return {
        "sample_id": str(events[0].get("sample_id", "")),
        "history": np.asarray(history, dtype=np.float32),
        "history_mask": np.ones(history_steps, dtype=np.float32),
        "future_dt": np.asarray(future_dt, dtype=np.float32),
        "future_pos": np.asarray(future_pos, dtype=np.float32),
        "future_motion": future_motion_arr,
        "anchor_latlon": np.asarray([lat_ref, lon_ref], dtype=np.float32),
    }


def build_sliding_windows(
    events: Sequence[Dict],
    history_steps: int,
    future_steps: int,
    window_stride: int,
    target_mode: str,
    min_future_dt: float,
    max_abs_velocity: float,
    max_abs_displacement: float,
) -> List[Dict]:
    total_needed = history_steps + future_steps
    if len(events) < total_needed:
        return []

    stride = max(int(window_stride), 1)
    windows: List[Dict] = []
    max_start = len(events) - total_needed
    for start in range(0, max_start + 1, stride):
        sub_events = events[start : start + total_needed]
        item = build_window_features(
            sub_events,
            history_steps,
            future_steps,
            target_mode,
            min_future_dt,
            max_abs_velocity,
            max_abs_displacement,
        )
        if item:
            base_id = str(sub_events[0].get("sample_id", ""))
            item["sample_id"] = f"{base_id}::start_{start}"
            windows.append(item)
    return windows


def fit_scale(motions: np.ndarray, stat: str) -> np.ndarray:
    flat = motions.reshape(-1, motions.shape[-1])
    if stat == "p95":
        scale = np.percentile(np.abs(flat), 95, axis=0)
    elif stat == "std":
        scale = np.std(flat, axis=0)
    else:
        raise ValueError(f"Unsupported scale stat: {stat}")
    return np.maximum(scale.astype(np.float32), EPS)


def fit_history_stats(items: List[Dict], stat: str) -> Dict[str, List[float]]:
    history = np.stack([item["history"] for item in items], axis=0)
    pos_scale = fit_scale(history[..., [HISTORY_X_IDX, HISTORY_Y_IDX]], stat)
    speed_scale = fit_scale(history[..., [HISTORY_SPEED_IDX]], stat)
    dt_log = np.log1p(np.maximum(history[..., [HISTORY_DT_IDX]], 0.0))
    dt_scale = fit_scale(dt_log, stat)
    return {
        "xy_scale": pos_scale.tolist(),
        "speed_scale": speed_scale.tolist(),
        "dt_scale": dt_scale.tolist(),
    }


def normalize_history(history: np.ndarray, history_stats: Dict) -> np.ndarray:
    out = history.copy().astype(np.float32)
    xy_scale = np.asarray(history_stats["xy_scale"], dtype=np.float32)
    speed_scale = np.asarray(history_stats["speed_scale"], dtype=np.float32)
    dt_scale = np.asarray(history_stats["dt_scale"], dtype=np.float32)
    out[..., HISTORY_X_IDX] /= xy_scale[0]
    out[..., HISTORY_Y_IDX] /= xy_scale[1]
    out[..., HISTORY_SPEED_IDX] /= speed_scale[0]
    out[..., HISTORY_DT_IDX] = np.log1p(np.maximum(out[..., HISTORY_DT_IDX], 0.0)) / dt_scale[0]
    out[..., [HISTORY_X_IDX, HISTORY_Y_IDX, HISTORY_SPEED_IDX, HISTORY_DT_IDX]] = np.clip(
        out[..., [HISTORY_X_IDX, HISTORY_Y_IDX, HISTORY_SPEED_IDX, HISTORY_DT_IDX]],
        -HISTORY_CLIP_VALUE,
        HISTORY_CLIP_VALUE,
    )
    return out


def dump_split(output_prefix: Path, items: List[Dict], target_mode: str, scale: np.ndarray, meta: Dict) -> None:
    history = np.stack([item["history"] for item in items], axis=0)
    history_norm = normalize_history(history, meta["history_stats"])
    history_mask = np.stack([item["history_mask"] for item in items], axis=0)
    future_dt = np.stack([item["future_dt"] for item in items], axis=0)
    future_pos = np.stack([item["future_pos"] for item in items], axis=0)
    future_motion = np.stack([item["future_motion"] for item in items], axis=0)
    sample_ids = np.asarray([item["sample_id"] for item in items])
    anchor_latlon = np.stack([item["anchor_latlon"] for item in items], axis=0)

    position_scale = np.asarray(meta["position_scale"], dtype=np.float32)

    target_npz = output_prefix.with_suffix(".npz")
    with tempfile.NamedTemporaryFile(dir=target_npz.parent, suffix=".npz", delete=False) as tmp_f:
        tmp_npz = Path(tmp_f.name)

    np.savez_compressed(
        tmp_npz,
        history=history_norm,
        history_raw=history,
        history_mask=history_mask,
        future_dt=future_dt,
        future_pos=future_pos,
        future_pos_norm=(future_pos / position_scale.reshape(1, 1, -1)).astype(np.float32),
        future_motion=(future_motion / scale.reshape(1, 1, -1)).astype(np.float32),
        sample_ids=sample_ids,
        anchor_latlon=anchor_latlon,
        target_mode=np.asarray([target_mode]),
    )
    tmp_npz.replace(target_npz)
    output_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = []
    with input_path.open("r") as f:
        for line in f:
            obj = json.loads(line)
            events = merge_events(obj, source_mode=args.source_mode)
            for event in events:
                event["sample_id"] = obj.get("sample_id", "")
            windows = build_sliding_windows(
                events,
                args.history_steps,
                args.future_steps,
                args.window_stride,
                args.target_mode,
                args.min_future_dt,
                args.max_abs_velocity,
                args.max_abs_displacement,
            )
            if windows:
                items.extend(windows)

    if not items:
        raise RuntimeError("No valid windows were produced. Adjust history/future lengths.")

    random.shuffle(items)

    if args.max_train_items and args.max_train_items > 0:
        if not (0.0 < args.train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0, 1) when max_train_items is used.")
        capped_total = int(math.ceil(float(args.max_train_items) / float(args.train_ratio)))
        if capped_total < 3:
            raise ValueError("max_train_items is too small to produce train/val/test splits.")
        if len(items) > capped_total:
            items = items[:capped_total]

    n_total = len(items)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val :]

    if not train_items or not val_items or not test_items:
        raise RuntimeError("Split produced an empty partition. Adjust split ratios or dataset size.")

    train_motion_physical = np.stack([item["future_motion"] for item in train_items], axis=0)
    motion_scale = fit_scale(train_motion_physical, args.scale_stat)
    train_pos_physical = np.stack([item["future_pos"] for item in train_items], axis=0)
    position_scale = fit_scale(train_pos_physical, args.scale_stat)
    history_stats = fit_history_stats(train_items, args.scale_stat)

    meta = {
        "input_jsonl": str(input_path),
        "source_mode": args.source_mode,
        "history_steps": args.history_steps,
        "future_steps": args.future_steps,
        "window_stride": args.window_stride,
        "target_mode": args.target_mode,
        "max_train_items": args.max_train_items,
        "feature_order": ["x", "y", "speed_knots", "course_sin", "course_cos", "dt", "source_id"],
        "history_feature_transform": {
            "x": "x / history_stats.xy_scale[0]",
            "y": "y / history_stats.xy_scale[1]",
            "speed_knots": "speed / history_stats.speed_scale[0]",
            "course_sin": "identity",
            "course_cos": "identity",
            "dt": "log1p(dt) / history_stats.dt_scale[0]",
            "source_id": "identity",
        },
        "history_clip_value": HISTORY_CLIP_VALUE,
        "history_stats": history_stats,
        "motion_scale": motion_scale.tolist(),
        "position_scale": position_scale.tolist(),
        "scale_stat": args.scale_stat,
        "min_future_dt": args.min_future_dt,
        "max_abs_velocity": args.max_abs_velocity,
        "max_abs_displacement": args.max_abs_displacement,
        "splits": {"train": len(train_items), "val": len(val_items), "test": len(test_items)},
        "note": (
            "Single-stream preprocessing disables the other AIS source entirely when source_mode is traj_a/traj_b. "
            "Small-dt and extreme-target windows are filtered for stability."
        ),
    }

    dump_split(output_dir / "train", train_items, args.target_mode, motion_scale, meta)
    dump_split(output_dir / "val", val_items, args.target_mode, motion_scale, meta)
    dump_split(output_dir / "test", test_items, args.target_mode, motion_scale, meta)
    (output_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
