#!/usr/bin/env bash
# Block extra training folds on one node with a shared pool of route workers: every fold's
# (target source x route group) task goes into one queue, longest groups first, PARALLEL processes at a time,
# so all cores stay busy. A fold-0 pilot runs first in the same pool as the platform check; afterwards each
# fold is finalized (shard hashes verified, fold<k>_metrics.json + fold<k>_manifest.json written).
# Shards go to artifacts/phase1c_fold<k>/work (the layout R4/k2_export expects). Resumable: rerun to continue.
# Usage (from code/business_entity_resolution): FOLDS="2 3 4" PARALLEL=8 bash cloud/run_folds_pool.sh
set -euo pipefail
. "$HOME/venv/bin/activate"
FOLDS=${FOLDS:?e.g. FOLDS="2 3 4"}
PARALLEL=${PARALLEL:-8}
DATA=${DATA:-$HOME/data}
LOGS="$DATA/logs"
mkdir -p "$LOGS"
TASKS="$DATA/fold_tasks.txt"
echo "pilot|0|-|-" > "$TASKS"
for group in address_char name_char "name_word exact_name rare_name suffix_exact"; do
  for k in $FOLDS; do
    for source in 3 2; do echo "fold|$k|$source|$group" >> "$TASKS"; done
  done
done

run_task() {
  local kind k source group
  IFS='|' read -r kind k source group <<< "$1"
  if [ "$kind" = "pilot" ]; then
    python -m src.evaluate_phase1c --config ../../configs/phase1c_fold0.json --scope pilot \
      --work-dir "$DATA/platform_check/work" --output-dir ../../artifacts/platform_check > "$LOGS/platform_check.log" 2>&1
    return
  fi
  local args=()
  for route in $group; do args+=(--only-route "$route"); done
  python -m src.evaluate_phase1c --config "../../configs/phase1c_fold$k.json" --scope full --extra-training-fold \
    --work-dir "../../artifacts/phase1c_fold$k/work" --output-dir "../../artifacts/phase1c_fold$k" \
    --only-source "$source" "${args[@]}" > "$LOGS/fold${k}_S${source}_${group%% *}.log" 2>&1
  echo "$(date +%H:%M) done: fold $k S$source $group"
}
export -f run_task
export DATA LOGS
echo "$(date +%H:%M) pool start: folds $FOLDS, $PARALLEL workers, $(wc -l < "$TASKS") tasks"
xargs -a "$TASKS" -d '\n' -P "$PARALLEL" -I{} bash -c 'run_task "$1"' _ {}

python - <<'PY'
import json, sys
node = json.load(open("../../artifacts/platform_check/pilot_reproduction.json"))
kaggle = json.load(open("../../artifacts/phase1c_kaggle/pilot_reproduction.json"))
identical = node["comparisons"] == kaggle["comparisons"]
print(f"platform check: basis={node['basis']} full_run_allowed={node['full_run_allowed']} identical_to_kaggle={identical}")
if not node["full_run_allowed"]:
    sys.exit("Platform check FAILED: candidates on this node are outside the platform tolerance; do not use these folds.")
PY
for k in $FOLDS; do
  python -m src.evaluate_phase1c --config "../../configs/phase1c_fold$k.json" --scope full --extra-training-fold \
    --work-dir "../../artifacts/phase1c_fold$k/work" --output-dir "../../artifacts/phase1c_fold$k" > "$LOGS/fold${k}_finalize.log" 2>&1
  tail -n 1 "$LOGS/fold${k}_finalize.log"
done
echo "$(date +%H:%M) pool complete"
