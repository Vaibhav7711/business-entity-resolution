#!/usr/bin/env bash
# Reranker test scoring on both Kaggle accounts: the 13.86M test cascade rows in 12 shards over 4 kernels (2 per
# account, 3 shards each; ~620 rows/s per 2xT4 kernel), then merge, check and join onto the augmented lists as the
# reranker features for test (rr_on_aug/test.npz). Markers: RR12_TEST_READY, or RR12_FAILED: <step>.
# Run on the node from code/business_entity_resolution.
set -uo pipefail
. "$HOME/venv/bin/activate"
D=$HOME/data; RR=$D/rerank; SHARDS=12; U1=vaibhav0383; U2=$(tr -d '[:space:]' < "$HOME/.kaggle_user2")
T1=$HOME/.kaggle_api_token; T2=$HOME/.kaggle_api_token2
fail() { echo "RR12_FAILED: $*"; exit 1; }
as1() { KAGGLE_API_TOKEN=$(cat "$T1") KAGGLE_TOKEN_FILE=$T1 KAGGLE_OWNER=$U1 "$@"; }
as2() { KAGGLE_API_TOKEN=$(cat "$T2") KAGGLE_TOKEN_FILE=$T2 KAGGLE_OWNER=$U2 KAGGLE_STAGE=$D/kaggle_stage2 "$@"; }
wait_kernel() {  # wait_kernel <owner/slug> <token file>: 0 complete, 1 failed or unknown for 10 polls
  local s blind=0
  while true; do
    s=$(KAGGLE_API_TOKEN=$(cat "$2") kaggle kernels status "$1" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
    echo "$(date -u +%H:%M) $1: ${s:-unknown}"
    case "$s" in *COMPLETE) return 0;; *ERROR|*CANCEL*) return 1;; "") blind=$((blind + 1)); [ $blind -ge 10 ] && return 1;; *) blind=0;; esac
    sleep 180
  done
}
ds_upload2() {  # ds_upload2 <slug> <dir> on account 2; fails on a reported error
  printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "other"}]}' "$1" "$U2" "$1" > "$2/dataset-metadata.json"
  local out
  if as2 kaggle datasets status "$U2/$1" > /dev/null 2>&1; then out=$(as2 kaggle datasets version -p "$2" -m update --dir-mode zip -q 2>&1) || true
  else out=$(as2 kaggle datasets create -p "$2" --dir-mode zip -q 2>&1) || true; fi
  echo "upload $1: ${out:-ok}"; ! grep -qi error <<< "$out"
}

# 1. Replace the two 6-shard kernels (3.1 h each at the measured rate).
N=after_rr2; P=$(pgrep -f "cloud/$N.sh"); [ -n "$P" ] && kill $P && echo "stopped after_rr2 ($P)"
for k in ber-rr-test-a ber-rr-test-b; do as1 kaggle kernels delete -y "$U1/$k" 2>&1 | tail -1; done

# 2. Account 1: two kernels (model from the ber-rr-train output, pairs ber-rr-pairs).
for part in a:0,1,2 b:3,4,5; do
  as1 env KERNEL_ENV="{\"CE_TEST_SHARDS\": \"$SHARDS\", \"CE_TEST_ONLY\": \"${part#*:}\"}" bash cloud/kaggle_ce.sh push \
    "ber-rr-t12-${part%%:*}" ce_rerank_t4.json score test "$U1/ber-ce-code,$U1/ber-rr-pairs" "$U1/ber-rr-train" \
    || fail "push ber-rr-t12-${part%%:*}"
  sleep 20
done

# 3. Account 2: the reranker model and the pairs as its own datasets, then two kernels.
rm -rf "$D/rr_model_dl" && as1 kaggle kernels output "$U1/ber-rr-train" -p "$D/rr_model_dl" \
  --file-pattern "(model/.*|train_log\.json)$" -o > /dev/null || fail "download the reranker model"
M=$(dirname "$(find "$D/rr_model_dl" -name train_log.json | head -1)")
[ -f "$M/model/config.json" ] || fail "no reranker model in the ber-rr-train output"
S=$D/kaggle_stage2/rr_model; rm -rf "$S" && mkdir -p "$S" && cp -al "$M/." "$S/"
ds_upload2 ber-rr-model "$S" || fail "upload ber-rr-model"
S=$D/kaggle_stage2/rr_pairs; rm -rf "$S" && mkdir -p "$S/notes" && cp -al "$D/pairs_cascade/test" "$S/test"
cp -al "$D/pairs_cascade/validation" "$S/validation"
echo "Round-1 top-8 cascade pairs (validation, test) for reranker scoring." > "$S/notes/README.txt"
ds_upload2 ber-rr-pairs "$S" || fail "upload ber-rr-pairs"
until as2 kaggle datasets files "$U2/ber-rr-model" --page-size 50 2>/dev/null | grep -q "train_log.json"; do sleep 20; done
until as2 kaggle datasets files "$U2/ber-rr-pairs" --page-size 50 2>/dev/null | grep -q "test/s1.parquet"; do sleep 20; done
sleep 60
for part in c:6,7,8 d:9,10,11; do
  as2 env KERNEL_ENV="{\"CE_TEST_SHARDS\": \"$SHARDS\", \"CE_TEST_ONLY\": \"${part#*:}\"}" bash cloud/kaggle_ce.sh push \
    "ber-rr-t12-${part%%:*}" ce_rerank_t4.json score test "$U2/ber-ce-code,$U2/ber-rr-pairs,$U2/ber-rr-model" - \
    || fail "push ber-rr-t12-${part%%:*}"
  sleep 20
done
echo "$(date -u +%H:%M) RR12_PUSHED"

# 4. Collect, merge, check, join.
sleep 300
for k in a b; do wait_kernel "$U1/ber-rr-t12-$k" "$T1" || fail "ber-rr-t12-$k"; done
for k in c d; do wait_kernel "$U2/ber-rr-t12-$k" "$T2" || fail "ber-rr-t12-$k"; done
rm -f "$RR"/scores/test.*
for k in a b c d; do
  owner=$U1; tok=$T1; case $k in c|d) owner=$U2; tok=$T2;; esac
  rm -rf "$D/rr12_$k" && KAGGLE_API_TOKEN=$(cat "$tok") kaggle kernels output "$owner/ber-rr-t12-$k" -p "$D/rr12_$k" \
    --file-pattern "(shard.*\.npy|[a-z_]+\.json|[a-z_]+\.log)$" -o > /dev/null || fail "download ber-rr-t12-$k"
  find "$D/rr12_$k" -name "test.shard*of$SHARDS.npy" -exec cp {} "$RR/scores/" \;
done
python -c "from pathlib import Path; from src.ce_model import merge_shards; import sys; sys.exit(0 if merge_shards(Path.home() / 'data/rerank/scores', 'test', $SHARDS) else 1)" \
  || fail "test shards incomplete ($(ls "$RR"/scores/test.shard* 2>/dev/null | wc -l)/$SHARDS)"
python cloud/check_scores.py --scores "$RR/scores" --rows-json "$D/pairs_cascade/cascade_summary.json" --splits test \
  || fail "merged reranker test scores"
python -m src.ce_join --src-root "$D/pairs_cascade" --src-scores "$RR/scores" --dst-root "$D/pairs_aug" \
  --out "$D/rr_on_aug" --name rr --splits test || fail "join the reranker test scores"
echo RR12_TEST_READY
