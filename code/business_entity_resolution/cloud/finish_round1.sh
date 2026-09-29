#!/usr/bin/env bash
# Submission A (run on the node, from code/business_entity_resolution): once round-1 test scoring is done, merge the
# test scores with round 1's validation/holdout scores, run the full policy (tune -> one owner -> both TSVs -> official
# validator), then build the test cascade (top-N by round 1) for a possible round-2 test run.
set -uo pipefail
. "$HOME/venv/bin/activate"
export KAGGLE_API_TOKEN=$(cat "$HOME/.kaggle_api_token")
DATA=${DATA:-$HOME/data}
while true; do
  s=$(kaggle kernels status vaibhav0383/ber-ce-test 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
  echo "$(date +%H:%M) ber-ce-test: $s"
  case "$s" in *COMPLETE) break;; *ERROR|*CANCEL*) echo "test scoring failed"; exit 1;; esac
  sleep 120
done
kaggle kernels output vaibhav0383/ber-ce-test -p "$DATA/ce_test_out" --file-pattern '(scores/test\.npy|test_k\.json|score_log\.json)$' -o > /dev/null
T=$(find "$DATA/ce_test_out" -name test.npy -path '*scores*' | head -1)
cp "$T" "$DATA/ce_out/ce/scores/test.npy"
cp "$(dirname "$(dirname "$T")")/test_k.json" "$DATA/ce_out/ce/test_k.json" 2>/dev/null || true
python -m src.ce_policy --pairs-root "$DATA/pairs_ce1" --scores-dir "$DATA/ce_out/ce/scores" \
  --test-dir ../../student_resource/dataset/test --train-dir ../../student_resource/dataset/train \
  --out "$DATA/policy_r1" --stage all
echo "SUBMISSION A READY: $DATA/policy_r1/output"; head -c 1200 "$DATA/policy_r1/POLICY_REPORT.md"
python -m src.ce_cascade --pairs-root "$DATA/pairs_ce1" --scores-dir "$DATA/ce_out/ce/scores" --out "$DATA/pairs_cascade" --n 8
echo "CASCADE READY"
