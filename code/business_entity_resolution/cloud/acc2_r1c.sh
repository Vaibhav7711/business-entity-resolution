#!/usr/bin/env bash
# Second Kaggle account (token ~/.kaggle_api_token2, username ~/.kaggle_user2), run on the node from
# code/business_entity_resolution:
#   1. dense lists for the fold-0 training S1 (the trained bi-encoder, attached as a dataset; test retrieval skipped);
#   2. union training pairs = fold-0 top-40 lists + dense top-20 additions (labels from the training gold);
#   3. round 1c = round 1b warm-started on the union pairs; scores the dense-route pairs of validation/holdout;
#   4. two kernels score the dense-route test pairs in shards (0-2, 3-5 of 6);
#   5. merged scores -> ~/data/r1c_new/scores/{validation,holdout,test}.npy for cloud/finish_dense_c.sh.
set -uo pipefail
. "$HOME/venv/bin/activate"
until [ -s "$HOME/.kaggle_api_token2" ] && [ -s "$HOME/.kaggle_user2" ]; do sleep 30; done
U2=$(tr -d '[:space:]' < "$HOME/.kaggle_user2")
export KAGGLE_API_TOKEN=$(cat "$HOME/.kaggle_api_token2")
export KAGGLE_TOKEN_FILE=$HOME/.kaggle_api_token2 KAGGLE_OWNER=$U2 KAGGLE_STAGE=$HOME/data/kaggle_stage2
D=$HOME/data; TR=../../student_resource/dataset/train; SHARDS=6
mkdir -p "$KAGGLE_STAGE"
echo "$(date -u +%H:%M) account 2: $U2"

ds_upload() {  # ds_upload <slug> <dir>: new dataset or new version; fails on a reported error (the CLI exits 0)
  printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}' "$1" "$U2" "$1" > "$2/dataset-metadata.json"
  local out
  if kaggle datasets status "$U2/$1" > /dev/null 2>&1; then
    out=$(kaggle datasets version -p "$2" -m "update" --dir-mode zip -q 2>&1) || true
  else
    out=$(kaggle datasets create -p "$2" --dir-mode zip -q 2>&1) || true
  fi
  echo "upload $1: ${out:-ok}"
  ! grep -qi error <<< "$out"
}
ds_wait() {  # ds_wait <slug> <file pattern>
  until kaggle datasets files "$U2/$1" --page-size 200 2>/dev/null | grep -q "$2"; do sleep 20; done
}
wait_kernel() {  # wait_kernel <slug>: 0 complete, 1 failed
  while true; do
    local s
    s=$(kaggle kernels status "$U2/$1" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
    echo "$(date -u +%H:%M) $1: $s"
    case "$s" in *COMPLETE) return 0;; *ERROR|*CANCEL*) return 1;; esac
    sleep 120
  done
}
fail() { echo "FAILED: $*"; exit 1; }
stage_dir() {  # stage_dir <name>: an empty staging directory
  rm -rf "${KAGGLE_STAGE:?}/$1" && mkdir -p "$KAGGLE_STAGE/$1" && echo "$KAGGLE_STAGE/$1"
}

# 1. Inputs for account 2 (its datasets are private to it).
bash cloud/kaggle_ce.sh code || fail "code upload"
bash cloud/kaggle_ce.sh fold0 || fail "fold0 upload"
BI=$(dirname "$(find "$D/bi_model_dl" -name train_log.json | head -1)")
R1B=$(dirname "$(find "$D/r1b_model_dl" -name train_log.json | head -1)")
S=$(stage_dir bi_model); cp -al "$BI/." "$S/"; ds_upload ber-bi-model "$S" || fail "bi model upload"
S=$(stage_dir r1b_model); cp -al "$R1B/." "$S/"; ds_upload ber-r1b-model "$S" || fail "r1b model upload"
S=$(stage_dir dense_new); mkdir -p "$S/notes"
for s in validation holdout test; do cp -al "$D/dense_new/$s" "$S/$s"; done
echo "Dense-route pairs not in the top-40 lists (dense rank < 20)." > "$S/notes/README.txt"
ds_upload ber-dense-new "$S" || fail "dense-new upload"
SIZE=$(stat -c %s src/ce_model.py)
until kaggle datasets files "$U2/ber-ce-code" --page-size 200 2>/dev/null | grep "src/ce_model.py" | grep -q " $SIZE "; do sleep 20; done
ds_wait ber-ce-pairs-fold0 "train/s1.parquet"; ds_wait ber-bi-model "train_log.json"; ds_wait ber-r1b-model "train_log.json"
ds_wait ber-dense-new "test/s1.parquet"
sleep 60
echo "$(date -u +%H:%M) account 2 inputs ready"

# 2. Dense lists for the fold-0 training S1.
KERNEL=bi_kernel.py bash cloud/kaggle_ce.sh push ber-bi-train bi_train.json all - \
  "$U2/ber-ce-code,$U2/ber-ce-pairs-fold0,$U2/ber-bi-model" || fail "push ber-bi-train"
sleep 90
wait_kernel ber-bi-train || { kaggle kernels output "$U2/ber-bi-train" -p "$D/bi_train_fail" -o > /dev/null; fail "ber-bi-train"; }
rm -rf "$D/bi_train_out" && kaggle kernels output "$U2/ber-bi-train" -p "$D/bi_train_out" \
  --file-pattern "(dense/train/.*|[a-z_]+\.json|[a-z_]+\.log)$" -o > /dev/null
DT=$(dirname "$(find "$D/bi_train_out" -path "*dense/train" -type d | head -1)")
[ -n "$DT" ] && [ -d "$DT/train" ] || fail "no dense/train lists"
python -m src.dense_merge --stage new --base-root "$D/pairs_fold0" --dense-root "$DT" --new-root "$D/dense_new_train" \
  --train-dir "$TR" --max-rank 20 --splits train || fail "dense_merge train"
cat "$D/dense_new_train/dense_merge_new.json"; echo

# 3. Union training pairs and round 1c (warm start from round 1b), scoring the validation/holdout dense pairs.
S=$(stage_dir union_train); mkdir -p "$S/train" "$S/notes"
for p in "$D"/pairs_fold0/train/part-*.parquet; do ln "$p" "$S/train/$(basename "$p")"; done
i=10000; for p in "$D"/dense_new_train/train/part-*.parquet; do ln "$p" "$S/train/part-$i.parquet"; i=$((i + 1)); done
ln "$D/pairs_fold0/train/s1.parquet" "$S/train/s1.parquet"
echo "Round-1c training pairs: fold-0 top-40 lists (part-0*) + dense route top-20 additions (part-1*)." > "$S/notes/README.txt"
ls -la "$S/train"
ds_upload ber-union-train "$S" || fail "union upload"
ds_wait ber-union-train "train/part-10000.parquet"
sleep 60
WARM_START=1 bash cloud/kaggle_ce.sh push ber-r1c ce_r1c.json all validation,holdout \
  "$U2/ber-ce-code,$U2/ber-union-train,$U2/ber-dense-new,$U2/ber-r1b-model" || fail "push ber-r1c"
sleep 90
wait_kernel ber-r1c || { kaggle kernels output "$U2/ber-r1c" -p "$D/r1c_fail" --file-pattern "(json|log)$" -o > /dev/null; fail "ber-r1c"; }

# 4. Test shards of the dense pairs on two kernels.
for part in a:0,1,2 b:3,4,5; do
  KERNEL_ENV="{\"CE_TEST_SHARDS\": \"$SHARDS\", \"CE_TEST_ONLY\": \"${part#*:}\"}" bash cloud/kaggle_ce.sh push \
    "ber-r1c-test-${part%%:*}" ce_r1c.json score test "$U2/ber-ce-code,$U2/ber-dense-new" "$U2/ber-r1c" || fail "push test ${part%%:*}"
  sleep 30
done
rm -rf "$D/r1c_out" && kaggle kernels output "$U2/ber-r1c" -p "$D/r1c_out" \
  --file-pattern "(scores/[a-z]+\.npy|[a-z_]+\.json|[a-z_]+\.log)$" -o > /dev/null
OUT=$(dirname "$(find "$D/r1c_out" -name train_log.json | head -1)")
python3 -c "import json,sys; t=json.load(open(sys.argv[1])); print({k: t.get(k) for k in ('pairs','pairs_seen','steps','throughput','stopped_early','warm_start_from','license')})" "$OUT/train_log.json"
cat "$OUT/eval.json" 2>/dev/null; echo
mkdir -p "$D/r1c_new/scores" && cp "$OUT/scores/validation.npy" "$OUT/scores/holdout.npy" "$D/r1c_new/scores/"
echo "R1C_VALIDATION_SCORES_READY"
sleep 120
for p in a b; do wait_kernel "ber-r1c-test-$p" || echo "ber-r1c-test-$p failed"; done
for p in a b; do
  rm -rf "$D/r1c_test_$p" && kaggle kernels output "$U2/ber-r1c-test-$p" -p "$D/r1c_test_$p" \
    --file-pattern "(shard.*\.npy|[a-z_]+\.json|[a-z_]+\.log)$" -o > /dev/null
  find "$D/r1c_test_$p" -name "test.shard*of$SHARDS.npy" -exec cp {} "$D/r1c_new/scores/" \;
done
python -c "from pathlib import Path; from src.ce_model import merge_shards; import sys; sys.exit(0 if merge_shards(Path.home() / 'data/r1c_new/scores', 'test', $SHARDS) else 1)" \
  || fail "r1c test shards incomplete"
echo "R1C_TEST_SCORES_READY"
