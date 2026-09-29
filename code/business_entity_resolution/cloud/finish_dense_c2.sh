#!/usr/bin/env bash
# Dense route with round 1c (account 2) scoring the dense pairs, every step checked. Two variants, chosen on
# validation out-of-fold macro F0.5:
#   c: pairs_aug_c = the augmented lists with round-1c logits on the dense rows (round 1b on the base rows);
#   x: pairs_aug (round 1b everywhere) + the round-1c dense-row logit as extra features (ce_join; NaN on base rows),
#      so one logit column never mixes two models' scales.
# Markers: DENSE_C_VALIDATION_DONE best=<variant>, DENSE_C_TEST_DONE; DENSE_C_FAILED: <step> otherwise.
set -uo pipefail
. "$HOME/venv/bin/activate"
D=$HOME/data; TE=../../student_resource/dataset/test; ROWS=$D/dense_new/dense_merge_new.json
fail() { echo "DENSE_C_FAILED: $*"; exit 1; }
blocked() { grep -q "FAILED" "$HOME/acc2_r1c.log" 2>/dev/null || grep -q "DENSE_FAILED" "$HOME/dense_pipeline2.log" 2>/dev/null; }
until grep -q R1C_VALIDATION_SCORES_READY "$HOME/acc2_r1c.log" && grep -q DENSE_VALIDATION_DONE "$HOME/dense_pipeline2.log"; do
  blocked && fail "an upstream chain failed"; sleep 60
done
python cloud/check_scores.py --scores "$D/r1c_new/scores" --rows-json "$ROWS" --splits validation holdout \
  || fail "round-1c validation/holdout scores do not match the dense pairs"
DENSE=$(dirname "$(find "$D/bi_out" -path "*dense/validation" -type d | head -1)")
R1B=$(dirname "$(find "$D/ce_r1b" -name validation.npy -path "*scores*" | head -1)")
AUG="--base-root $D/pairs_ce1 --dense-root $DENSE --new-root $D/dense_new --base-scores $R1B --new-scores $D/r1c_new/scores --aug-root $D/pairs_aug_c"
rm -rf "$D/pairs_aug_c" "$D/r1c_on_aug" "$D/stack_dense_c_val" "$D/stack_dense_x_val" "$D/stack_dense_c" "$D/stack_dense_x"
python -m src.dense_merge --stage augment $AUG --splits validation holdout || fail "augment c (validation/holdout)"
python -m src.ce_stack --pairs-root "$D/pairs_aug_c" --scores-dir "$D/pairs_aug_c/scores" --context-dir "$D/ctx_aug" \
  --extra-dir "$D/pairs_aug_c/extra" --out "$D/stack_dense_c_val" --no-test || fail "stack c (validation)"
python -m src.ce_join --src-root "$D/dense_new" --src-scores "$D/r1c_new/scores" --dst-root "$D/pairs_aug" \
  --out "$D/r1c_on_aug" --name r1c --splits validation holdout || fail "join x (validation/holdout)"
python -m src.ce_stack --pairs-root "$D/pairs_aug" --scores-dir "$D/pairs_aug/scores" --context-dir "$D/ctx_aug" \
  --extra-dir "$D/pairs_aug/extra" "$D/r1c_on_aug" --out "$D/stack_dense_x_val" --no-test || fail "stack x (validation)"
BEST=$(python cloud/best_report.py "$D/stack_dense_c_val" "$D/stack_dense_x_val" "$D/stack_dense_val")
echo "DENSE_C_VALIDATION_DONE best=$BEST"
case "$BEST" in stack_dense_c_val|stack_dense_x_val) ;; *) echo "round 1c does not beat the round-1b dense stack: no test stack"; exit 0;; esac
until grep -q R1C_TEST_SCORES_READY "$HOME/acc2_r1c.log"; do blocked && fail "account-2 test scoring failed"; sleep 60; done
python cloud/check_scores.py --scores "$D/r1c_new/scores" --rows-json "$ROWS" --splits test \
  || fail "round-1c test scores do not match the dense pairs"
if [ "$BEST" = stack_dense_c_val ]; then
  python -m src.dense_merge --stage augment $AUG --splits test || fail "augment c (test)"
  python -m src.ce_stack --pairs-root "$D/pairs_aug_c" --scores-dir "$D/pairs_aug_c/scores" --context-dir "$D/ctx_aug" \
    --extra-dir "$D/pairs_aug_c/extra" --test-dir "$TE" --out "$D/stack_dense_c" || fail "stack c (test)"
  OUTDIR=$D/stack_dense_c
else
  python -m src.ce_join --src-root "$D/dense_new" --src-scores "$D/r1c_new/scores" --dst-root "$D/pairs_aug" \
    --out "$D/r1c_on_aug" --name r1c --splits test || fail "join x (test)"
  python -m src.ce_stack --pairs-root "$D/pairs_aug" --scores-dir "$D/pairs_aug/scores" --context-dir "$D/ctx_aug" \
    --extra-dir "$D/pairs_aug/extra" "$D/r1c_on_aug" --test-dir "$TE" --out "$D/stack_dense_x" || fail "stack x (test)"
  OUTDIR=$D/stack_dense_x
fi
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r['test']['validator']['passed'] else 1)" \
  "$OUTDIR/stack_report.json" || fail "validator"
echo "DENSE_C_TEST_DONE $OUTDIR"
