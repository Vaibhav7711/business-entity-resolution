#!/usr/bin/env bash
# After round 1b (run on the node, from code/business_entity_resolution): policy on round 1b alone and on the
# round-1 + round-1b logit average; the variant with the best validation macro F0.5 (and at least +0.0003 over round 1)
# becomes the final candidate.
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
until [ -f "$DATA/policy_r1/policy_report.json" ] && [ -f "$DATA/ce_out/ce/scores/test.npy" ]; do sleep 60; done
kaggle kernels output vaibhav0383/ber-ce-r1b -p "$DATA/ce_r1b" --file-pattern "$PATTERN" -o > /dev/null
R1B=$(dirname "$(find "$DATA/ce_r1b" -name validation.npy -path '*scores*' | head -1)")
cat "$(dirname "$R1B")/train_log.json" | head -c 400; echo; cat "$(dirname "$R1B")/eval.json" | head -c 600; echo
COMMON="--pairs-root $DATA/pairs_ce1 --test-dir ../../student_resource/dataset/test --train-dir ../../student_resource/dataset/train --stage all"
python -m src.ce_policy --scores-dir "$R1B" --out "$DATA/policy_r1b" $COMMON
python -m src.ce_ensemble --scores-dirs "$DATA/ce_out/ce/scores" "$R1B" --out "$DATA/ce_ens"
python -m src.ce_policy --scores-dir "$DATA/ce_ens/scores" --out "$DATA/policy_ens" $COMMON
python - <<'PY'
import json
from pathlib import Path
data = Path.home() / "data"
rows = {}
for name in ("policy_r1", "policy_r1b", "policy_ens"):
    r = json.loads((data / name / "policy_report.json").read_text())
    c = r["chosen"]
    rows[name] = (c["validation"]["macro_f05"], c["holdout"]["macro_f05"], c["name"], r["test"]["validator"]["passed"])
    print(f"{name}: validation {rows[name][0]:.5f} holdout {rows[name][1]:.5f} ({rows[name][2]}), validator {rows[name][3]}")
best = max(rows, key=lambda k: rows[k][0])
if best != "policy_r1" and rows[best][0] < rows["policy_r1"][0] + 0.0003:
    best = "policy_r1"
print(f"FINAL CANDIDATE: {best} -> {data / best / 'output'}")
(data / "final_choice.txt").write_text(best + "\n")
PY
