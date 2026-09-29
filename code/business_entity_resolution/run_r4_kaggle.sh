#!/usr/bin/env bash
# R4 on Kaggle CPU: stage-2 matcher features for FOLDS (e.g. FOLDS="0 1"), exported compactly for R5.
# Inputs attached: this code, the training TSVs (satwiksps dataset), the fold-0 Phase 1C notebook output
# (always), and the fold-blocking notebook output of every extra fold listed in FOLDS.
# Output: /kaggle/working/r4/fold<k>/chunkNNN.npz + index.json (about 3-5 GB per fold).
set -euo pipefail
export PHASE2A_THREADS=1
FOLDS=${FOLDS:?set FOLDS, e.g. FOLDS="0 1"}
CONFIG=../../configs/k2_matcher.json
P1C=$(dirname "$(find /kaggle/input -name state.json -path '*phase1c/work*' | head -1)")
K1=$(dirname "$(find /kaggle/input -name k1_results.json | head -1)")
WORK=${WORK:-/tmp/k2_work}
OUT=${OUT:-/kaggle/working/r4}
mkdir -p "$OUT"
echo "fold-0 phase1c work: $P1C"; echo "k1 results: $K1"
test -f "$P1C/state.json" || { echo "attach the fold-0 Phase 1C notebook output"; exit 1; }
test -f "$K1/models/filter_final.txt"
for k in $FOLDS; do
  if [ "$k" != "0" ]; then
    M=$(find /kaggle/input -name "fold${k}_manifest.json" | head -1)
    [ -n "$M" ] || { echo "fold $k blocking output is not attached"; exit 1; }
    ln -sfn "$(dirname "$M")" "../../artifacts/phase1c_fold$k"
    echo "fold $k blocking: $(dirname "$M")"
  fi
done
python3 -m pip install -q -r requirements.txt
python3 -m pytest -q
python3 -m src.k2_export --config "$CONFIG" --phase1c-work-dir "$P1C" --k1-dir "$K1" --work-dir "$WORK" \
  --output-dir "$OUT" --folds $FOLDS 2>&1 | tee "$OUT/r4_run_${FOLDS// /_}.log" \
  | grep --line-buffered -E "R4|features: all|stores: complete|Error|Traceback|stopping"
du -sh "$OUT"/*
