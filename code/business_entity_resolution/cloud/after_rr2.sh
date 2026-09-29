#!/usr/bin/env bash
# Reranker (bge-reranker-v2-m3 on the round-1 top-8 cascade, Kaggle ber-rr-train), every step checked.
# Gate: if the dense stack exists, the reranker logits must improve it as extra features (ce_join onto pairs_aug) by
# at least GAIN on validation out-of-fold macro F0.5; otherwise (dense failed) a cascade stack must beat C. Only then
# are the test shards scored (two Kaggle kernels) and the test stack built with the official validator.
# Markers: RERANK_VALIDATION_DONE mode=... best=..., RERANK_TEST_DONE <dir>; RERANK_FAILED: <step> otherwise.
set -uo pipefail
. "$HOME/venv/bin/activate"
export KAGGLE_API_TOKEN=$(cat "$HOME/.kaggle_api_token")
D=$HOME/data; RR=$D/rerank; TE=../../student_resource/dataset/test; C_OOF=0.98080; GAIN=0.0002; SHARDS=6
CASCADE_ROWS=$D/pairs_cascade/cascade_summary.json
fail() { echo "RERANK_FAILED: $*"; exit 1; }
wait_kernel() {  # 0 complete, 1 failed or unknown for 10 polls
  local s blind=0
  while true; do
    s=$(kaggle kernels status "vaibhav0383/$1" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
    echo "$(date -u +%H:%M) $1: ${s:-unknown}"
    case "$s" in *COMPLETE) return 0;; *ERROR|*CANCEL*) return 1;; "") blind=$((blind + 1)); [ $blind -ge 10 ] && return 1;; *) blind=0;; esac
    sleep 180
  done
}
oof() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['validation_oof_macro_f05'])" "$1/stack_report.json"; }

wait_kernel ber-rr-train || fail "ber-rr-train"
rm -rf "$D/rr_train" && kaggle kernels output vaibhav0383/ber-rr-train -p "$D/rr_train" \
  --file-pattern "(scores/[a-z]+\.npy|[a-z_]+\.json|[a-z_]+\.log)$" -o > /dev/null || fail "download ber-rr-train"
OUT=$(dirname "$(find "$D/rr_train" -name train_log.json | head -1)")
[ -f "$OUT/train_log.json" ] || fail "no train_log.json in the ber-rr-train output"
python3 -c "import json,sys; t=json.load(open(sys.argv[1])); print({k: t.get(k) for k in ('pairs','positives','pairs_seen','steps','planned_steps','fitted_steps','throughput','stopped_early','final_window_loss','backbone','license')})" "$OUT/train_log.json"
cat "$OUT/eval.json"; echo
python cloud/check_scores.py --scores "$OUT/scores" --rows-json "$CASCADE_ROWS" --splits validation holdout \
  || fail "reranker scores do not match the cascade rows"
python3 - "$OUT/eval.json" "$CASCADE_ROWS" <<'PY' || fail "the kernel scored a different cascade than ~/data/pairs_cascade"
import json, sys
ev, cs = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bad = [s for s in ("validation", "holdout") if ev[s]["rows"] != cs[s]["rows"] or ev[s]["positives"] != cs[s]["positives"]]
print("cascade alignment:", "OK" if not bad else f"MISMATCH in {bad}")
sys.exit(1 if bad else 0)
PY
rm -rf "$RR" "$D/stack_rr_val" "$D/stack_rr_r1b_val" "$D/rr_on_aug" "$D/stack_dense_rr_val" "$D/stack_dense_rr" "$D/stack_rr"
mkdir -p "$RR/scores" && cp "$OUT/scores/validation.npy" "$OUT/scores/holdout.npy" "$RR/scores/"
CAS="--pairs-root $D/pairs_cascade --scores-dir $RR/scores --context-dir $D/ctx_cascade"
python -m src.ce_stack $CAS --out "$D/stack_rr_val" --no-test || fail "stack rr (validation)"
python -m src.ce_stack $CAS --extra-dir "$D/r1b_on_cascade" --out "$D/stack_rr_r1b_val" --no-test || fail "stack rr+r1b (validation)"
until grep -qE "DENSE_VALIDATION_DONE|DENSE_FAILED" "$HOME/dense_pipeline2.log" 2>/dev/null; do sleep 60; done
if grep -q DENSE_VALIDATION_DONE "$HOME/dense_pipeline2.log"; then
  MODE=dense
  python -m src.ce_join --src-root "$D/pairs_cascade" --src-scores "$RR/scores" --dst-root "$D/pairs_aug" \
    --out "$D/rr_on_aug" --name rr --splits validation holdout || fail "join rr onto the dense lists"
  python -m src.ce_stack --pairs-root "$D/pairs_aug" --scores-dir "$D/pairs_aug/scores" --context-dir "$D/ctx_aug" \
    --extra-dir "$D/pairs_aug/extra" "$D/rr_on_aug" --out "$D/stack_dense_rr_val" --no-test || fail "stack dense+rr (validation)"
  BAR=$(python3 -c "print($(oof "$D/stack_dense_val") + $GAIN)")
  BEST=$(python cloud/best_report.py "$D/stack_dense_rr_val" "$D/stack_dense_val" --beat "$BAR")
else
  MODE=cascade
  BEST=$(python cloud/best_report.py "$D/stack_rr_val" "$D/stack_rr_r1b_val" --beat "$C_OOF")
fi
python cloud/best_report.py "$D/stack_rr_val" "$D/stack_rr_r1b_val" > /dev/null
echo "RERANK_VALIDATION_DONE mode=$MODE best=$BEST"
case "$BEST" in stack_dense_rr_val|stack_rr_val|stack_rr_r1b_val) ;; *) echo "reranker adds nothing on validation: no test scoring"; exit 0;; esac

for part in a:0,1,2 b:3,4,5; do
  KERNEL_ENV="{\"CE_TEST_SHARDS\": \"$SHARDS\", \"CE_TEST_ONLY\": \"${part#*:}\"}" bash cloud/kaggle_ce.sh push \
    "ber-rr-test-${part%%:*}" ce_rerank_t4.json score test "vaibhav0383/ber-ce-code,vaibhav0383/ber-rr-pairs" \
    "vaibhav0383/ber-rr-train" || fail "push ber-rr-test-${part%%:*}"
  sleep 30
done
sleep 120
for p in a b; do wait_kernel "ber-rr-test-$p" || fail "ber-rr-test-$p"; done
rm -f "$RR"/scores/test.*
for p in a b; do
  rm -rf "$D/rr_test_$p" && kaggle kernels output "vaibhav0383/ber-rr-test-$p" -p "$D/rr_test_$p" \
    --file-pattern "(shard.*\.npy|[a-z_]+\.json|[a-z_]+\.log)$" -o > /dev/null || fail "download ber-rr-test-$p"
  find "$D/rr_test_$p" -name "test.shard*of$SHARDS.npy" -exec cp {} "$RR/scores/" \;
done
python -c "from pathlib import Path; from src.ce_model import merge_shards; import sys; sys.exit(0 if merge_shards(Path.home() / 'data/rerank/scores', 'test', $SHARDS) else 1)" \
  || fail "test shards incomplete"
python cloud/check_scores.py --scores "$RR/scores" --rows-json "$CASCADE_ROWS" --splits test || fail "merged test scores"
if [ "$BEST" = stack_dense_rr_val ]; then
  until grep -qE "DENSE_TEST_DONE|DENSE_FAILED" "$HOME/dense_pipeline2.log"; do sleep 60; done
  python -m src.ce_join --src-root "$D/pairs_cascade" --src-scores "$RR/scores" --dst-root "$D/pairs_aug" \
    --out "$D/rr_on_aug" --name rr --splits test || fail "join rr (test)"
  python -m src.ce_stack --pairs-root "$D/pairs_aug" --scores-dir "$D/pairs_aug/scores" --context-dir "$D/ctx_aug" \
    --extra-dir "$D/pairs_aug/extra" "$D/rr_on_aug" --test-dir "$TE" --out "$D/stack_dense_rr" || fail "stack dense+rr (test)"
  OUTDIR=$D/stack_dense_rr
else
  EXTRA=""; [ "$BEST" = stack_rr_r1b_val ] && EXTRA="--extra-dir $D/r1b_on_cascade"
  python -m src.ce_stack $CAS $EXTRA --test-dir "$TE" --out "$D/stack_rr" || fail "stack rr (test)"
  OUTDIR=$D/stack_rr
fi
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r['test']['validator']['passed'] else 1)" \
  "$OUTDIR/stack_report.json" || fail "validator"
echo "RERANK_TEST_DONE $OUTDIR"
