#!/usr/bin/env bash
# K2 on Kaggle CPU: stage-2 matcher experiments on the full fold-0 candidate universe.
# Inputs attached to the notebook: this code, the Phase 1C notebook output (fold-0 shards),
# the K1 results dataset (filter models, k1_results.json, substitutions), and the training TSVs.
# Large intermediates go to /tmp (not saved); results go to /kaggle/working/k2.
set -euo pipefail
export PHASE2A_THREADS=1
CONFIG=../../configs/k2_matcher.json
P1C=$(dirname "$(find /kaggle/input -name state.json -path '*phase1c/work*' | head -1)")
K1=$(dirname "$(find /kaggle/input -name k1_results.json | head -1)")
WORK=${WORK:-/tmp/k2_work}
OUT=${OUT:-/kaggle/working/k2}
mkdir -p "$OUT"
echo "phase1c work: $P1C"; echo "k1 results: $K1"
test -f "$P1C/state.json" && test -f "$K1/models/filter_final.txt"
python3 -m pip install -q -r requirements.txt
python3 -m pytest -q
python3 -m src.k2_experiments --config "$CONFIG" --phase1c-work-dir "$P1C" --k1-dir "$K1" \
  --work-dir "$WORK" --output-dir "$OUT" --stage all 2>&1 | tee "$OUT/k2_run.log" \
  | grep --line-buffered -E "K2:|stores:|features: all|E[0-4]|champion|learning curve|transfer|Error|Traceback|stopping"
cp "$WORK/canon_map.json" "$OUT/models/" 2>/dev/null || true
ls -la "$OUT" "$OUT/models"
