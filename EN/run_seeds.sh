#!/usr/bin/env bash
# Run the 5 English Full-model main-table seeds (SpeechOcean762).
# Seeds: 173, 185, 79, 237, 239
#
# Usage (from LAPG/EN):
#   CUDA_DEVICE=0 ./run_seeds.sh
#   CUDA_DEVICE=0 ./run_seeds.sh 173 185
set -euo pipefail

# Stay relative to this script (LAPG/EN)
cd "$(dirname "$0")"
PY="${PY:-python3}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEFAULT_SEEDS=(173 185 79 237 239)

run_one() {
  local SEED="$1"
  local OUT="exp/seed${SEED}"
  local LOG="${OUT}/train.out"
  mkdir -p "$OUT"

  if [[ -f "${OUT}/best_mdd_f1.pt" ]] && [[ -f "$LOG" ]] && grep -q '\[done\]' "$LOG" 2>/dev/null; then
    echo "[skip] seed=${SEED} already done → ${OUT}"
    return 0
  fi

  echo "======== [start] seed=${SEED} gpu=${CUDA_DEVICE} out=${OUT} ========"
  : >"${OUT}/train_log.jsonl"
  rm -rf "${OUT}/mdd_eval" "${OUT}/"*.pt 2>/dev/null || true

  "$PY" -u ./train_speechocean2.py \
    --phone-feat gop --word-feat gop --ssl-fuse film --qwen-fusion film \
    --film-order qwen_gop_ssl \
    --apa-phone-loss mse --apa-word-loss mse \
    --w-mdd 3 --w-detect 1 --w-apa 2 --w-utt 0 --apa-pcc-weight 0.35 \
    --joint-phone-weight 1.0 --joint-margin 0.2 \
    --phone-vocab ./resource/vocab_merge.json --best-metric mdd_f1 \
    --epochs 40 --early-stop-patience 10 --early-stop-min-epochs 12 \
    --batch-size 8 --num-workers 2 --lr 4e-4 \
    --seed "${SEED}" \
    --cuda-device "${CUDA_DEVICE}" \
    --out-dir "${OUT}" \
    >"$LOG" 2>&1

  echo "======== [done] seed=${SEED} ========"
  grep -E '\[done\]|best mdd_f1' "$LOG" | tail -5 || true
}

echo "[plan] EN Full model | cwd=$(pwd) | cuda=${CUDA_DEVICE} | seeds=${*:-${DEFAULT_SEEDS[*]}}"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=("${DEFAULT_SEEDS[@]}")
fi

for SEED in "${SEEDS[@]}"; do
  run_one "$SEED"
done
echo "[all-done] finished"
