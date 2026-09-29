#!/usr/bin/env bash
# K2 then K3 in one Kaggle commit, so the first submission needs no manual step between them.
# Inputs attached: this code, the training/test TSVs (satwiksps dataset), the Phase 1C fold-0 notebook
# output, the K1 results dataset, and the three test-blocking notebook outputs.
# K3 starts only when all three test ranges are attached and enough of the 12 h session is left, and it
# is capped by `timeout`, so a K3 overrun ends the commit normally and the K2 outputs are still saved.
set -uo pipefail
START=$(date +%s)
LIMIT=${LIMIT_SECONDS:-41400}      # 11.5 h of the 12 h session; the rest covers setup and saving outputs
MIN_K3=${MIN_K3_SECONDS:-10800}    # do not start K3 with less than 3 h left
bash run_k2_kaggle.sh || { echo "K2 failed; K3 not started"; exit 1; }
RANGES=$(find /kaggle/input -name "test_range_*_manifest.json" | wc -l)
LEFT=$((LIMIT - ($(date +%s) - START)))
echo "K2 finished after $(( ($(date +%s) - START) / 60 )) min; test ranges attached: $RANGES; seconds left: $LEFT"
if [ "$RANGES" -lt 3 ]; then echo "K3 not started: needs all three test-blocking outputs"; exit 0; fi
if [ "$LEFT" -lt "$MIN_K3" ]; then echo "K3 not started: too little session time left"; exit 0; fi
K2=/kaggle/working/k2 timeout --kill-after=60 "$LEFT" bash run_k3_kaggle.sh
STATUS=$?
if [ "$STATUS" -eq 124 ]; then echo "K3 stopped at the session budget; K2 outputs are saved"; fi
if [ "$STATUS" -ne 0 ] && [ "$STATUS" -ne 124 ]; then echo "K3 failed (status $STATUS); see k3/k3_run.log"; fi
exit 0
