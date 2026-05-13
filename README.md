# Traj-PathFormer

Traj-PathFormer is a research codebase for AIS trajectory forecasting with
patch-based temporal modeling. The implementation focuses on irregular,
multi-source vessel trajectories and supports both single-trajectory forecasting
and pairwise cross-completion between two trajectory streams.

The repository contains the core model definitions, preprocessing utilities,
training entry points, evaluation scripts, and manuscript materials. Large
datasets, generated caches, checkpoints, logs, and experiment work directories
are intentionally excluded from version control.

## Highlights

- Patch-based encoders for irregular trajectory histories.
- Learnable time embeddings for historical observations and future queries.
- Motion-target forecasting in velocity or displacement space.
- Pairwise cross-completion protocol for dual AIS trajectory streams.
- Baseline training suite for compact comparisons against recurrent and
  graph-based forecasters.
- Evaluation scripts for trajectory error, completion quality, robustness, and
  table-ready protocol reports.

## Repository Layout

```text
.
├── models/                                  # Traj-PathFormer and baseline 
├── scripts/                                 # Utility scripts for analysis/figures
├── preprocess_motion_data.py                # Raw AIS JSONL to motion-target NPZ data
├── train_ours_singletraj_noslide.py         # Single-trajectory Traj-PathFormer training
├── train_pair_patch_cross_completion.py     # Pairwise cross-completion training
├── quick_singletraj_baseline_suite.py       # Baseline training suite
├── evaluate_*.py                            # Evaluation and robustness scripts
├── pair_patch_data.py                       # Pairwise data construction/cache utilities
├── motion_dataset.py                        # Processed single-trajectory dataset loader
└── trajectory_eval_utils.py                 # Shared trajectory metrics
```

## Environment

The code has been validated with Python 3.8 and PyTorch 2.x. Install the core
dependencies with:

```bash
pip install -r requirements.txt
```

For CUDA training, install a PyTorch build compatible with your local CUDA
driver before running the training scripts.

## Data

Raw data is expected as JSONL where each line contains trajectory features with
an `order` field and trajectory streams such as `traj_a` and `traj_b`.

Expected feature names include:

- `time`
- `lat`
- `lon`
- `vel`
- `cou`

Large raw files and generated `.npz` caches are not tracked. Place local data
under `data/` or pass explicit paths through command-line arguments.

## Preprocessing

Single-stream or merged-stream motion-target data can be generated with:

```bash
python preprocess_motion_data.py \
  --input-jsonl data/raw/dual_ais.jsonl \
  --output-dir data/singletraj_pair \
  --source-mode pair \
  --history-steps 64 \
  --future-steps 12 \
  --target-mode velocity
```

For one-stream experiments, set `--source-mode traj_a` or `--source-mode traj_b`.

## Training

Train the single-trajectory Traj-PathFormer model:

```bash
python train_ours_singletraj_noslide.py \
  --data-dir data/singletraj_pair \
  --workdir workdir_ours_singletraj_noslide \
  --device cuda:0 \
  --epochs 60 \
  --batch-size 64
```

Train the pairwise cross-completion model:

```bash
python train_pair_patch_cross_completion.py \
  --input-jsonl data/raw/dual_ais_outlierdrop200.jsonl \
  --workdir workdir_pair_cross_completion \
  --cache-dir workdir_pair_cross_completion/data_cache_rawpair \
  --device cuda:0 \
  --epochs 30 \
  --batch-size 64
```

The shell scripts `run_pair_patch_cross_completion_gpu.sh` and
`run_train_gpu2_bs1_e10.sh` provide reproducible launch templates. Override
`PYTHON_BIN`, `GPU_ID`, `INPUT_JSONL`, `BATCH_SIZE`, `EPOCHS`, and `LR` through
environment variables as needed.

## Evaluation

Evaluate a trained single-trajectory model:

```bash
python evaluate_trajectory_metrics.py \
  --exp-dir workdir_ours_singletraj_noslide/experiments/<experiment_name> \
  --data-dir data/singletraj_pair \
  --split test \
  --device cuda:0
```

Evaluate pairwise completion protocols:

```bash
python evaluate_table2_pair_protocols.py \
  --exp-dir workdir_pair_cross_completion/experiments/<experiment_name> \
  --cache-dir workdir_pair_cross_completion/data_cache_rawpair \
  --split test \
  --device cuda:0
```

Additional scripts report completion quality and mask robustness:

```bash
python evaluate_pair_patch_cross_completion_quality.py --help
python evaluate_pair_patch_cross_completion_mask_robustness.py --help
```

## Outputs

Training writes checkpoints, JSON logs, and summaries under the configured
`workdir`. These files are ignored by Git by default:

- `workdir*/`
- `*.pt`, `*.pth`, `*.ckpt`
- `*.log`, `*.out`, `*.pid`
- generated `.npz` caches

Keep those artifacts locally or publish them separately when needed for
reproducibility.

## License

Copyright (c) 2026 Yuting Song

All rights reserved.

This repository is publicly visible for academic review and reference only.

No permission is granted to copy, reproduce, modify, distribute, sublicense, publish,
or use the source code, dataset, documentation, models, or any part of this repository
for commercial or non-commercial purposes without explicit written permission from
the copyright holder.
