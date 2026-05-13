import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


EARTH_METERS_PER_LON = 111320.0
EARTH_METERS_PER_LAT = 110540.0
EPS = 1e-6
FEAT_X = 0
FEAT_Y = 1
FEAT_SPEED = 2
FEAT_COURSE_SIN = 3
FEAT_COURSE_COS = 4
FEAT_DT = 5
HISTORY_CLIP_VALUE = 5.0


def append_log(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def build_experiment_dir(workdir: Path, exp_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = workdir / "experiments" / f"{exp_name}_{stamp}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def log_device_info(device: torch.device) -> None:
    payload = {"device": str(device), "cuda_available": torch.cuda.is_available()}
    if device.type == "cuda":
        payload.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_name": torch.cuda.get_device_name(device),
            }
        )
    print(json.dumps({"device_info": payload}), flush=True)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def latlon_to_local_xy(lat: float, lon: float, lat_ref: float, lon_ref: float) -> Tuple[float, float]:
    lat_ref_rad = math.radians(lat_ref)
    x = (lon - lon_ref) * EARTH_METERS_PER_LON * math.cos(lat_ref_rad)
    y = (lat - lat_ref) * EARTH_METERS_PER_LAT
    return x, y


def read_events(obj: Dict, key: str) -> List[Dict]:
    order = obj["features"]["order"]
    time_idx = order.index("time")
    lat_idx = order.index("lat")
    lon_idx = order.index("lon")
    vel_idx = order.index("vel")
    cou_idx = order.index("cou")
    events = []
    for row in obj["features"].get(key, []):
        events.append(
            {
                "t": float(row[time_idx]),
                "lat": float(row[lat_idx]),
                "lon": float(row[lon_idx]),
                "speed": float(row[vel_idx]),
                "course": float(row[cou_idx]),
            }
        )
    events.sort(key=lambda x: x["t"])
    return events


def event_to_feature(event: Dict, prev_t: float, lat_ref: float, lon_ref: float) -> List[float]:
    x, y = latlon_to_local_xy(event["lat"], event["lon"], lat_ref, lon_ref)
    course_rad = math.radians(event["course"])
    return [
        x,
        y,
        event["speed"],
        math.sin(course_rad),
        math.cos(course_rad),
        max(event["t"] - prev_t, 0.0),
    ]


def build_time_patches(
    events: Sequence[Dict],
    anchor_t: float,
    lat_ref: float,
    lon_ref: float,
    npatch: int,
    patch_minutes: int,
    max_points_per_patch: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patch = np.zeros((npatch, max_points_per_patch, 6), dtype=np.float32)
    mask = np.zeros((npatch, max_points_per_patch), dtype=np.float32)
    hms = np.zeros((npatch, 3), dtype=np.float32)
    patch_start = np.zeros((npatch,), dtype=np.float32)
    patch_seconds = float(patch_minutes) * 60.0
    start0 = anchor_t - npatch * patch_seconds

    for patch_idx in range(npatch):
        start_t = start0 + patch_idx * patch_seconds
        end_t = start_t + patch_seconds
        patch_start[patch_idx] = start_t
        hms[patch_idx] = np.asarray(
            [
                (start_t - anchor_t) / 3600.0,
                ((start_t + end_t) * 0.5 - anchor_t) / 3600.0,
                (end_t - anchor_t) / 3600.0,
            ],
            dtype=np.float32,
        )
        pts = [event for event in events if start_t <= event["t"] < end_t]
        pts = pts[-max_points_per_patch:]
        prev_t = start_t
        for point_idx, event in enumerate(pts):
            patch[patch_idx, point_idx] = np.asarray(event_to_feature(event, prev_t, lat_ref, lon_ref), dtype=np.float32)
            mask[patch_idx, point_idx] = 1.0
            prev_t = event["t"]
    return patch, mask, hms, patch_start


def build_future_targets(
    future_events: Sequence[Dict],
    anchor_t: float,
    lat_ref: float,
    lon_ref: float,
    target_mode: str,
    min_future_dt: float,
    max_abs_velocity: float,
    max_abs_displacement: float,
) -> Dict:
    future_dt = []
    future_pos = []
    future_motion = []
    prev_t = anchor_t
    prev_x, prev_y = 0.0, 0.0
    for event in future_events:
        cur_x, cur_y = latlon_to_local_xy(event["lat"], event["lon"], lat_ref, lon_ref)
        dt = max(event["t"] - prev_t, 0.0)
        dx = cur_x - prev_x
        dy = cur_y - prev_y
        if target_mode == "velocity":
            if dt < min_future_dt:
                return {}
            motion = [dx / max(dt, EPS), dy / max(dt, EPS)]
            if max(abs(motion[0]), abs(motion[1])) > max_abs_velocity:
                return {}
        else:
            motion = [dx, dy]
            if max(abs(motion[0]), abs(motion[1])) > max_abs_displacement:
                return {}
        future_dt.append(dt)
        future_pos.append([cur_x, cur_y])
        future_motion.append(motion)
        prev_t = event["t"]
        prev_x, prev_y = cur_x, cur_y
    return {
        "future_dt": np.asarray(future_dt, dtype=np.float32),
        "future_pos": np.asarray(future_pos, dtype=np.float32),
        "future_motion": np.asarray(future_motion, dtype=np.float32),
        "anchor_latlon": np.asarray([lat_ref, lon_ref], dtype=np.float32),
    }


def build_pair_samples(args) -> List[Dict]:
    samples = []
    input_path = Path(args.input_jsonl)
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            anchor_events = read_events(obj, args.anchor_key)
            sender_events = read_events(obj, args.sender_key)
            if len(anchor_events) < args.min_anchor_points + args.future_steps or len(sender_events) < args.min_sender_points:
                continue

            max_start = len(anchor_events) - args.future_steps
            if args.no_sliding_window:
                future_starts = [max_start] if max_start >= args.min_anchor_points else []
            else:
                future_starts = range(args.min_anchor_points, max_start + 1, max(int(args.window_stride_points), 1))

            for future_start in future_starts:
                anchor_history = anchor_events[:future_start]
                future_events = anchor_events[future_start : future_start + args.future_steps]
                last_event = anchor_history[-1]
                anchor_t = last_event["t"]
                lat_ref = last_event["lat"]
                lon_ref = last_event["lon"]
                target = build_future_targets(
                    future_events,
                    anchor_t,
                    lat_ref,
                    lon_ref,
                    args.target_mode,
                    args.min_future_dt,
                    args.max_abs_velocity,
                    args.max_abs_displacement,
                )
                if not target:
                    continue
                anchor_patch_raw, anchor_mask, anchor_hms, anchor_start = build_time_patches(
                    anchor_history,
                    anchor_t,
                    lat_ref,
                    lon_ref,
                    args.npatch,
                    args.patch_minutes,
                    args.max_points_per_patch,
                )
                sender_patch_raw, sender_mask, sender_hms, sender_start = build_time_patches(
                    sender_events,
                    anchor_t,
                    lat_ref,
                    lon_ref,
                    args.npatch,
                    args.patch_minutes,
                    args.max_points_per_patch,
                )
                if anchor_mask.sum() < args.min_anchor_points or sender_mask.sum() < args.min_sender_points:
                    continue
                samples.append(
                    {
                        "sample_id": f"{obj.get('sample_id', '')}::anchor_{future_start}",
                        "anchor_patch_raw": anchor_patch_raw,
                        "anchor_mask": anchor_mask,
                        "anchor_patch_hms": anchor_hms,
                        "anchor_patch_start": anchor_start,
                        "sender_patch_raw": sender_patch_raw,
                        "sender_mask": sender_mask,
                        "sender_patch_hms": sender_hms,
                        "sender_patch_start": sender_start,
                        **target,
                    }
                )
                if args.max_raw_pairs and len(samples) >= args.max_raw_pairs:
                    return samples
    return samples


def fit_scale(values: np.ndarray, stat: str) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])
    if stat == "std":
        scale = np.std(flat, axis=0)
    else:
        scale = np.percentile(np.abs(flat), 95, axis=0)
    return np.maximum(scale.astype(np.float32), EPS)


def normalize_patch(patch: np.ndarray, stats: Dict[str, List[float]]) -> np.ndarray:
    out = patch.copy().astype(np.float32)
    xy_scale = np.asarray(stats["xy_scale"], dtype=np.float32)
    speed_scale = np.asarray(stats["speed_scale"], dtype=np.float32)
    dt_scale = np.asarray(stats["dt_scale"], dtype=np.float32)
    out[..., FEAT_X] /= xy_scale[0]
    out[..., FEAT_Y] /= xy_scale[1]
    out[..., FEAT_SPEED] /= speed_scale[0]
    out[..., FEAT_DT] = np.log1p(np.maximum(out[..., FEAT_DT], 0.0)) / dt_scale[0]
    out[..., [FEAT_X, FEAT_Y, FEAT_SPEED, FEAT_DT]] = np.clip(
        out[..., [FEAT_X, FEAT_Y, FEAT_SPEED, FEAT_DT]],
        -HISTORY_CLIP_VALUE,
        HISTORY_CLIP_VALUE,
    )
    return out


def fit_feature_stats(items: List[Dict], stat: str) -> Dict[str, List[float]]:
    patches = np.concatenate(
        [np.reshape(item["anchor_patch_raw"], (-1, 6)) for item in items]
        + [np.reshape(item["sender_patch_raw"], (-1, 6)) for item in items],
        axis=0,
    )
    xy_scale = fit_scale(patches[:, [FEAT_X, FEAT_Y]], stat)
    speed_scale = fit_scale(patches[:, [FEAT_SPEED]], stat)
    dt_scale = fit_scale(np.log1p(np.maximum(patches[:, [FEAT_DT]], 0.0)), stat)
    return {"xy_scale": xy_scale.tolist(), "speed_scale": speed_scale.tolist(), "dt_scale": dt_scale.tolist()}


def dump_pair_split(path: Path, items: List[Dict], meta: Dict) -> None:
    motion_scale = np.asarray(meta["motion_scale"], dtype=np.float32)
    position_scale = np.asarray(meta["position_scale"], dtype=np.float32)
    feature_stats = meta["feature_stats"]
    np.savez_compressed(
        path,
        sample_ids=np.asarray([item["sample_id"] for item in items]),
        anchor_patch=np.stack([normalize_patch(item["anchor_patch_raw"], feature_stats) for item in items], axis=0),
        anchor_patch_raw=np.stack([item["anchor_patch_raw"] for item in items], axis=0),
        anchor_mask=np.stack([item["anchor_mask"] for item in items], axis=0),
        anchor_patch_hms=np.stack([item["anchor_patch_hms"] for item in items], axis=0),
        anchor_patch_start=np.stack([item["anchor_patch_start"] for item in items], axis=0),
        sender_patch=np.stack([normalize_patch(item["sender_patch_raw"], feature_stats) for item in items], axis=0),
        sender_patch_raw=np.stack([item["sender_patch_raw"] for item in items], axis=0),
        sender_mask=np.stack([item["sender_mask"] for item in items], axis=0),
        sender_patch_hms=np.stack([item["sender_patch_hms"] for item in items], axis=0),
        sender_patch_start=np.stack([item["sender_patch_start"] for item in items], axis=0),
        future_dt=np.stack([item["future_dt"] for item in items], axis=0),
        future_pos=np.stack([item["future_pos"] for item in items], axis=0),
        future_pos_norm=(np.stack([item["future_pos"] for item in items], axis=0) / position_scale.reshape(1, 1, -1)).astype(np.float32),
        future_motion_norm=(np.stack([item["future_motion"] for item in items], axis=0) / motion_scale.reshape(1, 1, -1)).astype(np.float32),
        anchor_latlon=np.stack([item["anchor_latlon"] for item in items], axis=0),
        target_mode=np.asarray([meta["target_mode"]]),
    )
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_or_load_cache(args, cache_dir: Path) -> Dict:
    meta_path = cache_dir / "dataset_meta.json"
    required = [cache_dir / "train.npz", cache_dir / "val.npz", cache_dir / "test.npz", meta_path]
    if all(path.exists() for path in required):
        return json.loads(meta_path.read_text(encoding="utf-8"))

    cache_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    items = build_pair_samples(args)
    if not items:
        raise RuntimeError("No valid pair patch samples were produced.")
    random.shuffle(items)
    n_total = len(items)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val :]
    if not train_items or not val_items or not test_items:
        raise RuntimeError("Split produced an empty partition.")

    motion_scale = fit_scale(np.stack([item["future_motion"] for item in train_items], axis=0), args.scale_stat)
    position_scale = fit_scale(np.stack([item["future_pos"] for item in train_items], axis=0), args.scale_stat)
    meta = {
        "input_jsonl": str(Path(args.input_jsonl)),
        "anchor_key": args.anchor_key,
        "sender_key": args.sender_key,
        "npatch": args.npatch,
        "patch_minutes": args.patch_minutes,
        "max_points_per_patch": args.max_points_per_patch,
        "future_steps": args.future_steps,
        "no_sliding_window": bool(args.no_sliding_window),
        "window_stride_points": args.window_stride_points,
        "target_mode": args.target_mode,
        "scale_stat": args.scale_stat,
        "motion_scale": motion_scale.tolist(),
        "position_scale": position_scale.tolist(),
        "feature_stats": fit_feature_stats(train_items, args.scale_stat),
        "splits": {"train": len(train_items), "val": len(val_items), "test": len(test_items)},
    }
    dump_pair_split(cache_dir / "train.npz", train_items, meta)
    dump_pair_split(cache_dir / "val.npz", val_items, meta)
    dump_pair_split(cache_dir / "test.npz", test_items, meta)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class PairPatchDataset:
    def __init__(self, npz_path: str):
        self.path = Path(npz_path)
        blob = np.load(self.path, allow_pickle=False)
        self.sample_ids = blob["sample_ids"]
        self.anchor_patch = blob["anchor_patch"].astype(np.float32)
        self.anchor_patch_raw = blob["anchor_patch_raw"].astype(np.float32)
        self.anchor_mask = blob["anchor_mask"].astype(np.float32)
        self.anchor_patch_hms = blob["anchor_patch_hms"].astype(np.float32)
        self.anchor_patch_start = blob["anchor_patch_start"].astype(np.float32)
        self.sender_patch = blob["sender_patch"].astype(np.float32)
        self.sender_patch_raw = blob["sender_patch_raw"].astype(np.float32)
        self.sender_mask = blob["sender_mask"].astype(np.float32)
        self.sender_patch_hms = blob["sender_patch_hms"].astype(np.float32)
        self.sender_patch_start = blob["sender_patch_start"].astype(np.float32)
        self.future_dt = blob["future_dt"].astype(np.float32)
        self.future_pos = blob["future_pos"].astype(np.float32)
        self.future_pos_norm = blob["future_pos_norm"].astype(np.float32)
        self.future_motion_norm = blob["future_motion_norm"].astype(np.float32)
        self.anchor_latlon = blob["anchor_latlon"].astype(np.float32)
        self.target_mode = str(blob["target_mode"][0])

    def __len__(self) -> int:
        return int(self.anchor_patch.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "anchor_patch": self.anchor_patch[idx],
            "anchor_patch_raw": self.anchor_patch_raw[idx],
            "anchor_mask": self.anchor_mask[idx],
            "anchor_patch_hms": self.anchor_patch_hms[idx],
            "anchor_patch_start": self.anchor_patch_start[idx],
            "sender_patch": self.sender_patch[idx],
            "sender_patch_raw": self.sender_patch_raw[idx],
            "sender_mask": self.sender_mask[idx],
            "sender_patch_hms": self.sender_patch_hms[idx],
            "sender_patch_start": self.sender_patch_start[idx],
            "future_dt": self.future_dt[idx],
            "future_pos": self.future_pos[idx],
            "future_pos_norm": self.future_pos_norm[idx],
            "future_motion_norm": self.future_motion_norm[idx],
            "anchor_latlon": self.anchor_latlon[idx],
        }
