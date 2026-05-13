import json
from pathlib import Path
from typing import Dict

import numpy as np


class MotionDataset:
    def __init__(self, npz_path: str):
        self.path = Path(npz_path)
        blob = np.load(self.path, allow_pickle=False)
        self.history = blob["history"].astype(np.float32)
        self.history_raw = blob["history_raw"].astype(np.float32) if "history_raw" in blob.files else self.history.copy()
        self.history_mask = blob["history_mask"].astype(np.float32)
        self.future_dt = blob["future_dt"].astype(np.float32)
        self.future_pos = blob["future_pos"].astype(np.float32)
        self.future_pos_norm = blob["future_pos_norm"].astype(np.float32)
        self.future_motion = blob["future_motion"].astype(np.float32)
        self.sample_ids = blob["sample_ids"]
        self.anchor_latlon = blob["anchor_latlon"].astype(np.float32)
        self.target_mode = str(blob["target_mode"][0])

        meta_path = self.path.with_suffix(".meta.json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def __len__(self) -> int:
        return int(self.history.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return {
            "history": self.history[idx],
            "history_raw": self.history_raw[idx],
            "history_mask": self.history_mask[idx],
            "future_dt": self.future_dt[idx],
            "future_pos": self.future_pos[idx],
            "future_pos_norm": self.future_pos_norm[idx],
            "future_motion": self.future_motion[idx],
            "sample_id": self.sample_ids[idx],
            "anchor_latlon": self.anchor_latlon[idx],
        }
