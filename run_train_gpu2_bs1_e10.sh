#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKDIR="${ROOT_DIR}/workdir_pair_cross_completion"
LOG_DIR="${WORKDIR}/launch_logs"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/train_gpu2_bs1_e10_lr3e-5_rawpair_${STAMP}.log"
PID_FILE="${LOG_DIR}/train_gpu2_bs1_e10_lr3e-5_rawpair_${STAMP}.pid"
GPU_ID="${GPU_ID:-2}"
INPUT_JSONL="${INPUT_JSONL:-data/raw/dual_ais_level2_shape_aggressiveC_pointclean_ema_recomputed_outlierdrop200.jsonl}"

mkdir -p "${LOG_DIR}" "${WORKDIR}"
cd "${ROOT_DIR}"

setsid env CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_pair_patch_cross_completion.py \
  --device cuda:0 \
  --input-jsonl "${INPUT_JSONL}" \
  --no-sliding-window \
  --epochs 10 \
  --batch-size 1 \
  --lr 3e-5 \
  --num-workers 0 \
  --exp-name pair_patch_cross_completion_gpu2_bs1_e10_lr3e-5_rawpair \
  --workdir "${WORKDIR}" \
  --cache-dir "${WORKDIR}/data_cache_rawpair" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "${pid}" > "${PID_FILE}"
printf 'PID=%s\nLOG=%s\nPID_FILE=%s\n' "${pid}" "${LOG_FILE}" "${PID_FILE}"
