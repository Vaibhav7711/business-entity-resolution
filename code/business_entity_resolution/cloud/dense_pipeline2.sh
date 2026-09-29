#!/usr/bin/env bash
# Dense route, from the round-1b scoring of the dense pairs (Kaggle ber-dense-score) onward, with every step checked:
# download -> score gate -> augmented lists (top-38 scored base rows + dense rows) -> context -> stack (validation,
# then test with the official validator). Markers: DENSE_VALIDATION_DONE / DENSE_TEST_DONE only after the report
# exists (and the validator passed); DENSE_FAILED: <step> otherwise. Run on the node from code/business_entity_resolution.
set -uo pipefail
. "$HOME/venv/bin/activate"
export KAGGLE_API_TOKEN=$(cat "$HOME/.kaggle_api_token")
D=$HOME/data; TR=../../student_resource/dataset/train; TE=../../student_resource/dataset/test
fail() { echo "DENSE_FAILED: $*"; exit 1; }
wait_kernel() {  # 0 complete, 1 failed or unknown for 10 polls
  local s blind=0
  while true; do
    s=$(kaggle kernels status "vaibhav0383/$1" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
    echo "$(date -u +%H:%M) $1: ${s:-unknown}"
    case "$s" in *COMPLETE) return 0;; *ERROR|*CANCEL*) return 1;; "") blind=$((blind + 1)); [ $blind -ge 10 ] && return 1;; *) blind=0;; esac
    sleep 120
  done
}
DENSE=$(dirname "$(find "$D/bi_out" -path "*dense/validation" -type d | head -1)")
[ -d "$DENSE/validation" ] && [ -d "$DENSE/test" ] || fail "no dense lists under $D/bi_out"
R1B=$(dirname "$(find "$D/ce_r1b" -name validation.npy -path "*scores*" | head -1)")
[ -f "$R1B/test.npy" ] && [ -f "$(dirname "$R1B")/test_k.json" ] || fail "round-1b scores or test_k.json missing"
wait_kernel ber-dense-score || fail "ber-dense-score"
rm -rf "$D/dense_scores" && kaggle kernels output vaibhav0383/ber-dense-score -p "$D/dense_scores" \
  --file-pattern "(scores/[a-z]+\.npy|[a-z_]+\.json|[a-z_]+\.log)$" -o > /dev/null || fail "download ber-dense-score"
NEWS=$(dirname "$(find "$D/dense_scores" -name validation.npy -path "*scores*" | head -1)")
python cloud/check_scores.py --scores "$NEWS" --rows-json "$D/dense_new/dense_merge_new.json" \
  --splits validation holdout test || fail "dense-pair scores do not match the dense pairs"
rm -rf "$D/pairs_aug" "$D/ctx_aug" "$D/stack_dense_val" "$D/stack_dense"
python -m src.dense_merge --stage augment --base-root "$D/pairs_ce1" --dense-root "$DENSE" --new-root "$D/dense_new" \
  --base-scores "$R1B" --new-scores "$NEWS" --aug-root "$D/pairs_aug" || fail "augment"
cat "$D/pairs_aug/dense_merge_augment.json"; echo
python -m src.ce_context --pairs-root "$D/pairs_aug" --out "$D/ctx_aug" --train-dir "$TR" --test-dir "$TE" \
  --folds ../../artifacts/folds.tsv || fail "context"
python -m src.ce_stack --pairs-root "$D/pairs_aug" --scores-dir "$D/pairs_aug/scores" --context-dir "$D/ctx_aug" \
  --extra-dir "$D/pairs_aug/extra" --out "$D/stack_dense_val" --no-test || fail "validation stack"
[ -s "$D/stack_dense_val/stack_report.json" ] || fail "no validation report"
echo DENSE_VALIDATION_DONE
python -m src.ce_stack --pairs-root "$D/pairs_aug" --scores-dir "$D/pairs_aug/scores" --context-dir "$D/ctx_aug" \
  --extra-dir "$D/pairs_aug/extra" --test-dir "$TE" --out "$D/stack_dense" || fail "test stack (or validator)"
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r['test']['validator']['passed'] else 1)" \
  "$D/stack_dense/stack_report.json" || fail "validator"
echo DENSE_TEST_DONE
