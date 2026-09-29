#!/usr/bin/env bash
# Final candidate (run on the node, from code/business_entity_resolution): once round 1b is scored, run the context
# stacker on round 1b and on the round-1 + round-1b average (round 1 alone is ~/data/stack_r1), then pick the variant
# with the best validation out-of-fold macro F0.5 (holdout reported, never used to choose).
set -uo pipefail
. "$HOME/venv/bin/activate"
export KAGGLE_API_TOKEN=$(cat "$HOME/.kaggle_api_token")
DATA=${DATA:-$HOME/data}
PATTERN='(scores/[a-z]+\.npy|[a-z_]+\.json|[a-z_]+\.log)$'
while true; do
  s=$(kaggle kernels status vaibhav0383/ber-ce-r1b 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
  echo "$(date +%H:%M) ber-ce-r1b: $s"
  case "$s" in *COMPLETE) break;; *ERROR|*CANCEL*) echo "round 1b failed"; exit 1;; esac
  sleep 120
done
until [ -f "$DATA/stack_r1/stack_report.json" ]; do sleep 60; done
kaggle kernels output vaibhav0383/ber-ce-r1b -p "$DATA/ce_r1b" --file-pattern "$PATTERN" -o > /dev/null
R1B=$(dirname "$(find "$DATA/ce_r1b" -name validation.npy -path '*scores*' | head -1)")
head -c 500 "$(dirname "$R1B")/train_log.json"; echo; cat "$(dirname "$R1B")/eval.json" | head -c 500; echo
COMMON="--pairs-root $DATA/pairs_ce1 --context-dir $DATA/ctx --test-dir ../../student_resource/dataset/test"
python -m src.ce_stack --scores-dir "$R1B" --out "$DATA/stack_r1b" $COMMON
python -m src.ce_ensemble --scores-dirs "$DATA/ce_out/ce/scores" "$R1B" --out "$DATA/ce_ens"
python -m src.ce_stack --scores-dir "$DATA/ce_ens/scores" --out "$DATA/stack_ens" $COMMON
python - <<'PY'
import json
from pathlib import Path
data = Path.home() / "data"
rows = {}
for name in ("stack_r1", "stack_r1b", "stack_ens"):
    r = json.loads((data / name / "stack_report.json").read_text())
    rows[name] = (r["validation_oof_macro_f05"], r["holdout"]["macro_f05"], r["test"]["validator"]["passed"])
    print(f"{name}: validation(OOF) {rows[name][0]:.5f} holdout {rows[name][1]:.5f} validator {rows[name][2]}")
best = max(rows, key=lambda k: rows[k][0])
print(f"FINAL CANDIDATE: {best} -> {data / best / 'output'}")
(data / "final_choice.txt").write_text(best + "\n")
PY
