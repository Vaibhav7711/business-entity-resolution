#!/usr/bin/env bash
# Block one extra training fold (1-4) on Kaggle CPU with the frozen B+C+E blocker; one notebook per fold.
#   FOLD=1 bash run_extra_fold_kaggle.sh
# Six workers (target source x route group) as in the fold-0 run, then the finalize pass verifies every
# shard hash and writes fold<k>_metrics.json and fold<k>_manifest.json. Everything lands under
# artifacts/phase1c_fold<k>/ and is saved as the notebook output. These folds are training data only;
# validation and test candidates keep coming from the fold-0 and test-blocking runs.
set -euo pipefail
FOLD=${FOLD:?set FOLD to 1, 2, 3 or 4}
CONFIG=../../configs/phase1c_fold$FOLD.json
OUT=../../artifacts/phase1c_fold$FOLD
LOGS=$OUT/logs
mkdir -p "$LOGS"
python3 -m pip install -q -r requirements.txt
python3 -m pytest -q
pids=()
for source in 2 3; do
  for group in address_char name_char "name_word exact_name rare_name suffix_exact"; do
    route_args=(); for route in $group; do route_args+=(--only-route "$route"); done
    python3 -m src.evaluate_phase1c --config "$CONFIG" --scope full --extra-training-fold --work-dir "$OUT/work" \
      --output-dir "$OUT" --only-source "$source" "${route_args[@]}" > "$LOGS/worker_S${source}_${group%% *}.log" 2>&1 &
    pids+=($!)
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then tail -n 20 "$LOGS"/worker_*.log; echo "a shard worker failed; see $LOGS" >&2; exit 1; fi
python3 -m src.evaluate_phase1c --config "$CONFIG" --scope full --extra-training-fold --work-dir "$OUT/work" \
  --output-dir "$OUT" > "$LOGS/finalize.log" 2>&1
tail -n 5 "$LOGS/finalize.log"
ls -la "$OUT"
