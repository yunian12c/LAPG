#!/usr/bin/env bash
# Run the 5 best Chinese Full-model seeds (ranked by apa_mean_pcc).
# Seeds: 180, 265, 70, 226, 12
#
# Usage (from LAPG/CN):
#   CUDA_DEVICE=0 ./run_seeds.sh
#   CUDA_DEVICE=0 ./run_seeds.sh 180 265
set -euo pipefail

# Stay relative to this script (LAPG/CN)
cd "$(dirname "$0")"
PY="${PY:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEFAULT_SEEDS=(180 265 70 226 12)

run_one() {
  local SEED="$1"
  local OUT="exp/seed${SEED}"
  local LOG="${OUT}/train.out"
  mkdir -p "$OUT"

  if [[ -f "${OUT}/best_model.pt" ]] && [[ -f "$LOG" ]] && grep -qE '\[early stop\]|\[done\]|epoch 60/' "$LOG" 2>/dev/null; then
    echo "[skip] seed=${SEED} already done â†?${OUT}"
    return 0
  fi

  echo "======== [start] seed=${SEED} gpu=${CUDA_DEVICE} out=${OUT} ========"
  : >"${OUT}/train_log.jsonl"
  rm -f "${OUT}/"*.pt 2>/dev/null || true

  "$PY" -u ./train_graph.py \
    --qwen-fusion film \
    --cuda-device "${CUDA_DEVICE}" \
    --seed "${SEED}" \
    --out-dir "${OUT}" \
    >"$LOG" 2>&1

  echo "======== [done] seed=${SEED} ========"
  grep -E '\[early stop\]|apa_mean_pcc|best' "$LOG" | tail -5 || true
}

echo "[plan] CN Full model | cwd=$(pwd) | cuda=${CUDA_DEVICE} | seeds=${*:-${DEFAULT_SEEDS[*]}}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=("${DEFAULT_SEEDS[@]}")
fi

for SEED in "${SEEDS[@]}"; do
  run_one "$SEED"
done
echo "[all-done] finished"
