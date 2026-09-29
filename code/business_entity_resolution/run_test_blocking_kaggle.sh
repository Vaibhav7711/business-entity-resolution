#!/usr/bin/env bash
# Test-set candidate generation with the frozen, gate-validated B+C+E blocker.
# One shard range per Kaggle notebook (SHARDS=0-23 | 24-46 | 47-69); no labels are read.
# Six parallel workers (target source x route group), then a finalize pass that
# hash-verifies every task of the range and writes an unlabeled diagnostics manifest.
# Rerunning resumes: completed (source, country, route, shard) tasks are skipped.
set -euo pipefail
SHARDS=${SHARDS:?set SHARDS, e.g. SHARDS=0-23}
CONFIG=../../configs/phase1c_fold0.json
WORK_DIR=${WORK_DIR:-../../artifacts/test_blocking/work}
LOGS="$WORK_DIR/logs"
mkdir -p "$LOGS"

python3 -m pip install -q -r requirements.txt
python3 -m pytest -q

pids=()
for source in 2 3; do
  for group in address_char name_char "name_word exact_name rare_name suffix_exact"; do
    route_args=()
    for route in $group; do route_args+=(--only-route "$route"); done
    python3 -m src.block_test --config "$CONFIG" --shards "$SHARDS" --work-dir "$WORK_DIR" \
      --only-source "$source" "${route_args[@]}" > "$LOGS/worker_S${source}_${group%% *}_${SHARDS}.log" 2>&1 &
    pids+=($!)
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then
  echo "A worker failed; inspect $LOGS and rerun this script to resume." >&2
  exit 1
fi
python3 -m src.block_test --config "$CONFIG" --shards "$SHARDS" --work-dir "$WORK_DIR" --finalize \
  > "$LOGS/finalize_${SHARDS}.log" 2>&1
tail -n 3 "$LOGS/finalize_${SHARDS}.log"
