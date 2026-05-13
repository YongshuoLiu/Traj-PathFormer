#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKDIR="${ROOT_DIR}/workdir_pair_cross_completion"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
EXP_NAME="${EXP_NAME:-pair_patch_cross_completion_e30_lr1e4}"
INPUT_JSONL="${INPUT_JSONL:-data/raw/dual_ais_level2_shape_aggressiveC_pointclean_ema_recomputed_outlierdrop200.jsonl}"
NO_SLIDING_WINDOW="${NO_SLIDING_WINDOW:-0}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${WORKDIR}/launch_logs"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${STAMP}.log"

cd "${ROOT_DIR}"
mkdir -p "${WORKDIR}"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] starting pair patch cross completion"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "LOG_FILE=${LOG_FILE}"
echo "INPUT_JSONL=${INPUT_JSONL}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

extra_args=()
if [[ "${NO_SLIDING_WINDOW}" == "1" ]]; then
  extra_args+=(--no-sliding-window)
fi

"${PYTHON_BIN}" train_pair_patch_cross_completion.py \
  --device cuda:0 \
  --input-jsonl "${INPUT_JSONL}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --exp-name "${EXP_NAME}" \
  --workdir "${WORKDIR}" \
  "${extra_args[@]}" 2>&1 | tee "${LOG_FILE}"
