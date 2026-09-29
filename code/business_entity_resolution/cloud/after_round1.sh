#!/usr/bin/env bash
# Overnight orchestration after the round-1 cross-encoder kernel (run on the node, from code/business_entity_resolution):
# fetch round-1 scores -> policy + submission A files -> cascade pairs (top-N by round 1) -> upload -> once the round-2
# training kernel is done, push round-2 scoring -> fetch -> stacked policy + final submission files.
set -uo pipefail
. "$HOME/venv/bin/activate"
export KAGGLE_API_TOKEN=$(cat "$HOME/.kaggle_api_token")
DATA=${DATA:-$HOME/data}
TEST_DIR=../../student_resource/dataset/test
TRAIN_DIR=../../student_resource/dataset/train
PATTERN='(scores/[a-z]+\.npy|[a-z_]+\.json|[a-z_]+\.log)$'
wait_kernel() {  # wait_kernel <slug>: returns 0 on COMPLETE
  while true; do
    s=$(kaggle kernels status "vaibhav0383/$1" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
    echo "$(date +%H:%M) $1: $s"
    case "$s" in *COMPLETE) return 0;; *ERROR|*CANCEL*) return 1;; esac
    sleep 120
  done
}
wait_kernel ber-ce || { echo "round 1 failed"; exit 1; }
kaggle kernels output vaibhav0383/ber-ce -p "$DATA/ce_out" --file-pattern "$PATTERN" -o > /dev/null
S1=$(dirname "$(find "$DATA/ce_out" -name validation.npy -path '*scores*' | head -1)")
echo "round-1 scores: $S1"; cat "$(dirname "$S1")/eval.json" 2>/dev/null; cat "$(dirname "$S1")/test_k.json" 2>/dev/null
mkdir -p "$DATA/pairs_ce1"
for s in validation holdout; do ln -sfn "$DATA/pairs_fold0/$s" "$DATA/pairs_ce1/$s"; done
ln -sfn "$DATA/pairs_test/test" "$DATA/pairs_ce1/test"
python -m src.ce_policy --pairs-root "$DATA/pairs_ce1" --scores-dir "$S1" --test-dir "$TEST_DIR" --train-dir "$TRAIN_DIR" \
  --out "$DATA/policy_r1" --stage all
echo "=== submission A files: $DATA/policy_r1/output"; head -c 1500 "$DATA/policy_r1/POLICY_REPORT.md"
python -m src.ce_cascade --pairs-root "$DATA/pairs_ce1" --scores-dir "$S1" --out "$DATA/pairs_cascade" --retention 0.998 --max-n 10
bash cloud/kaggle_ce.sh cascade
for n in $(seq 1 60); do [ "$(kaggle datasets status vaibhav0383/ber-ce-pairs-cascade 2>&1)" = "ready" ] && break; sleep 30; done
wait_kernel ber-ce-large-train || { echo "round-2 training failed"; exit 1; }
bash cloud/kaggle_ce.sh push ber-ce-large-score ce_large.json score validation,holdout,test \
  vaibhav0383/ber-ce-code,vaibhav0383/ber-ce-pairs-cascade vaibhav0383/ber-ce-large-train
sleep 120
wait_kernel ber-ce-large-score || { echo "round-2 scoring failed"; exit 1; }
kaggle kernels output vaibhav0383/ber-ce-large-score -p "$DATA/ce2_out" --file-pattern "$PATTERN" -o > /dev/null
S2=$(dirname "$(find "$DATA/ce2_out" -name validation.npy -path '*scores*' | head -1)")
python -m src.ce_policy --pairs-root "$DATA/pairs_cascade" --scores-dir "$S2" --test-dir "$TEST_DIR" --train-dir "$TRAIN_DIR" \
  --out "$DATA/policy_r2" --stage all
echo "=== final candidate files: $DATA/policy_r2/output"; head -c 1500 "$DATA/policy_r2/POLICY_REPORT.md"
