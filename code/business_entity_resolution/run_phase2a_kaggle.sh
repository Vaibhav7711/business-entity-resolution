#!/usr/bin/env bash
# Phase 2A on Kaggle CPU, independent of the full Phase 1C run.
# 1) Regenerates ONLY benchmark candidates (fold-0 shards 0-2) with the frozen Phase 1C code
#    into its own work/output dirs (artifacts/phase1c/work_bench, artifacts/phase1c_bench).
# 2) Builds the Phase 2A pair dataset, features, QA; trains rules -> logistic -> LightGBM; writes the report.
# Rerunning resumes: completed shards/chunks/models are skipped where checkpointed.
set -euo pipefail
export PHASE2A_THREADS=${PHASE2A_THREADS:-4}
CONFIG1C=../../configs/phase1c_fold0.json
CONFIG2A=../../configs/phase2a_kaggle.json
LOGS=../../artifacts/phase2a/logs
mkdir -p "$LOGS"

python3 -m pip install -q -r requirements.txt
python3 -m pytest -q

python3 -m src.evaluate_phase1c --config "$CONFIG1C" --scope benchmark \
  --work-dir ../../artifacts/phase1c/work_bench --output-dir ../../artifacts/phase1c_bench \
  > "$LOGS/00_phase1c_benchmark.log" 2>&1
python3 - <<'PY'
import json, sys
r = json.load(open("../../artifacts/phase1c_bench/pilot_reproduction.json"))
print("benchmark candidates pilot basis:", r["basis"], json.dumps(r["platform_tolerance"]["deltas"]))
if not r["full_run_allowed"]:
    sys.exit("Benchmark candidates are outside the platform tolerance; stop before Phase 2A.")
PY

python3 -m src.phase2a_pairs --config "$CONFIG2A" --stage all > "$LOGS/01_pairs_texts_features_qa.log" 2>&1
for model in rules logistic lightgbm; do
  python3 -m src.phase2a_train --config "$CONFIG2A" --model "$model" > "$LOGS/02_train_${model}.log" 2>&1
  tail -n 1 "$LOGS/02_train_${model}.log"
done
python3 -m src.phase2a_report --config "$CONFIG2A" > /dev/null
echo "Phase 2A complete: artifacts/phase2a/README.md"
