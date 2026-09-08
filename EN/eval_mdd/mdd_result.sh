#!/bin/bash
# HMamba-style MDD eval: align-text + S/I/D analysis + compute-wer(PER).
# Usage: mdd_result.sh <human_seq> <ref> <hyp> <work_dir>
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <human-seq> <ref> <hyp> [work_dir]"
  echo "  human-seq : realized / human perceived phones"
  echo "  ref       : canonical phones"
  echo "  hyp       : model prediction"
  exit 1
fi

HUMAN_IN=$1
REF_IN=$2
HYP_IN=$3
WORK=${4:-$(pwd)}
mkdir -p "$WORK"
cd "$WORK"

ROOT="$(cd "$(dirname "$0")" && pwd)"
# PATH: use current environment

# keep originals, work on filtered copies in WORK
"$ROOT/utils/filter_scp.pl" -f 1 "$HYP_IN" "$HUMAN_IN" | sort -k1,1 > human_seq
"$ROOT/utils/filter_scp.pl" -f 1 "$HYP_IN" "$REF_IN" | sort -k1,1 > ref
sort -k1,1 "$HYP_IN" > hyp

align-text ark:ref ark:human_seq ark,t:- | "$ROOT/utils/wer_per_utt_details.pl" > ref_human_detail
align-text ark:human_seq ark:hyp ark,t:- | "$ROOT/utils/wer_per_utt_details.pl" > human_our_detail
align-text ark:ref ark:hyp ark,t:- | "$ROOT/utils/wer_per_utt_details.pl" > ref_our_detail

# analysis script expects cwd files named ref_human_detail / human_our_detail / ref_our_detail
# Prefer python from PATH (GNN env); fall back to python3.
if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi
$PY "$ROOT/utils/ins_del_sub_cor_analysis.py"

echo "===== PER (hyp vs realized) ====="
compute-wer --text --mode=present ark:human_seq ark:hyp
