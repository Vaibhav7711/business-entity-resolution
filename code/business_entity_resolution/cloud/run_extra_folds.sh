#!/usr/bin/env bash
# Block training folds 1-4 with the frozen, gate-validated blocker (x86 Intel node), all folds in parallel.
# 1) Platform check: fold-0 pilot on this node must pass the pilot gate (exact or within the platform tolerance).
# 2) Six workers per fold (source x route group); 3) finalize each fold -> fold<k>_metrics.json + fold<k>_manifest.json.
# Resumable: rerun after any interruption. Usage (from code/business_entity_resolution): bash cloud/run_extra_folds.sh
set -euo pipefail
. "$HOME/venv/bin/activate"
DATA=${DATA:-$HOME/data}
LOGS="$DATA/logs"; mkdir -p "$LOGS"
python -m src.evaluate_phase1c --config ../../configs/phase1c_fold0.json --scope pilot \
  --work-dir "$DATA/platform_check/work" --output-dir ../../artifacts/platform_check > "$LOGS/platform_check.log" 2>&1 &
check_pid=$!
pids=()
for k in 1 2 3 4; do
  for source in 2 3; do
    for group in address_char name_char "name_word exact_name rare_name suffix_exact"; do
      route_args=(); for route in $group; do route_args+=(--only-route "$route"); done
      python -m src.evaluate_phase1c --config "../../configs/phase1c_fold$k.json" --scope full --extra-training-fold \
        --work-dir "$DATA/folds/fold$k/work" --output-dir "../../artifacts/phase1c_fold$k" --only-source "$source" \
        "${route_args[@]}" > "$LOGS/fold${k}_S${source}_${group%% *}.log" 2>&1 &
      pids+=($!)
    done
  done
done
wait "$check_pid"
python - <<'PY'
import json, sys
node = json.load(open("../../artifacts/platform_check/pilot_reproduction.json"))
kaggle = json.load(open("../../artifacts/phase1c_kaggle/pilot_reproduction.json"))
identical = node["comparisons"] == kaggle["comparisons"]
print(f"platform check: basis={node['basis']} full_run_allowed={node['full_run_allowed']} identical_to_kaggle={identical}")
# Extra folds are training data only (validation and test candidates both come from Kaggle), so the
# documented platform tolerance is sufficient; a failure here means real drift and stops the run.
if not node["full_run_allowed"]:
    sys.exit("Platform check FAILED: candidates on this node are outside the platform tolerance; stop and investigate.")
PY
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "a fold worker failed; see $LOGS, rerun to resume"; exit 1; }
for k in 1 2 3 4; do
  python -m src.evaluate_phase1c --config "../../configs/phase1c_fold$k.json" --scope full --extra-training-fold \
    --work-dir "$DATA/folds/fold$k/work" --output-dir "../../artifacts/phase1c_fold$k" > "$LOGS/fold${k}_finalize.log" 2>&1 &
done
wait
for k in 1 2 3 4; do tail -n 1 "$LOGS/fold${k}_finalize.log"; done
