#!/usr/bin/env bash
# Dense-route pipeline (run on the node, from code/business_entity_resolution) after the bi-encoder kernel (ber-bi):
# new pairs -> Kaggle dataset -> cross-encoder scoring with the round-1 and round-1b models -> augmented lists/scores/
# extra features (r1, r1b, ensemble) -> context features -> stacker with the dense route. Selection by validation OOF.
set -uo pipefail
. "$HOME/venv/bin/activate"
export KAGGLE_API_TOKEN=$(cat "$HOME/.kaggle_api_token")
DATA=${DATA:-$HOME/data}
PATTERN='(scores/[a-z]+\.npy|[a-z_]+\.json|[a-z_]+\.log)$'
TEST_DIR=../../student_resource/dataset/test; TRAIN_DIR=../../student_resource/dataset/train
wait_kernel() {
  while true; do
    s=$(kaggle kernels status "vaibhav0383/$1" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
    echo "$(date +%H:%M) $1: $s"
    case "$s" in *COMPLETE) return 0;; *ERROR|*CANCEL*) return 1;; esac
    sleep 90
  done
}
wait_kernel ber-bi || { echo "bi-encoder kernel failed"; exit 1; }
kaggle kernels output vaibhav0383/ber-bi -p "$DATA/bi_out" -o > /dev/null
DENSE=$(dirname "$(find "$DATA/bi_out" -type d -name validation -path '*dense*' | head -1)")
cat "$(find "$DATA/bi_out" -name dense_report.json | head -1)"
python -m src.dense_merge --stage new --base-root "$DATA/pairs_ce1" --dense-root "$DENSE" --new-root "$DATA/pairs_new" \
  --train-dir "$TRAIN_DIR" --max-rank ${MAX_RANK:-20}
S=$DATA/kaggle_stage/pairs_new; rm -rf "$S" && mkdir -p "$S"
for s in validation holdout test; do cp -al "$DATA/pairs_new/$s" "$S/$s"; done
printf '{"title": "ber-ce-pairs-new", "id": "vaibhav0383/ber-ce-pairs-new", "licenses": [{"name": "other"}]}' > "$S/dataset-metadata.json"
kaggle datasets create -p "$S" --dir-mode zip -q || kaggle datasets version -p "$S" -m "dense route" --dir-mode zip -q
for n in $(seq 1 60); do
  [ "$(kaggle datasets status vaibhav0383/ber-ce-pairs-new 2>&1)" = "ready" ] && \
  [ "$(kaggle datasets files vaibhav0383/ber-ce-pairs-new --page-size 200 2>&1 | grep -c '^test/part-')" -ge 1 ] && break
  sleep 30
done
sleep 30
bash cloud/kaggle_ce.sh push ber-ce-new-r1 ce.json score validation,holdout,test vaibhav0383/ber-ce-code,vaibhav0383/ber-ce-pairs-new vaibhav0383/ber-ce
bash cloud/kaggle_ce.sh push ber-ce-new-r1b ce_r1b.json score validation,holdout,test vaibhav0383/ber-ce-code,vaibhav0383/ber-ce-pairs-new vaibhav0383/ber-ce-r1b
sleep 120
wait_kernel ber-ce-new-r1 || { echo "new-pair scoring (r1) failed"; exit 1; }
wait_kernel ber-ce-new-r1b || echo "new-pair scoring (r1b) failed; continuing with r1 only"
kaggle kernels output vaibhav0383/ber-ce-new-r1 -p "$DATA/ce_new_r1" --file-pattern "$PATTERN" -o > /dev/null
kaggle kernels output vaibhav0383/ber-ce-new-r1b -p "$DATA/ce_new_r1b" --file-pattern "$PATTERN" -o > /dev/null
NEW1=$(dirname "$(find "$DATA/ce_new_r1" -name test.npy -path '*scores*' | head -1)")
NEW1B=$(dirname "$(find "$DATA/ce_new_r1b" -name test.npy -path '*scores*' | head -1)")
BASE1B=$(dirname "$(find "$DATA/ce_r1b" -name test.npy -path '*scores*' | head -1)")
variants="r1:$DATA/ce_out/ce/scores:$NEW1"
[ -n "$NEW1B" ] && [ -n "$BASE1B" ] && variants="$variants r1b:$BASE1B:$NEW1B"
for v in $variants; do
  IFS=: read -r name base new <<< "$v"
  python -m src.dense_merge --stage augment --base-root "$DATA/pairs_ce1" --dense-root "$DENSE" --new-root "$DATA/pairs_new" \
    --base-scores "$base" --new-scores "$new" --aug-root "$DATA/aug_$name"
done
if [ -d "$DATA/aug_r1b" ]; then
  python -m src.ce_ensemble --scores-dirs "$DATA/aug_r1/scores" "$DATA/aug_r1b/scores" --out "$DATA/aug_ens_scores" --splits validation holdout test
  rm -f "$DATA/aug_ens_scores/test_k.json"
fi
python -m src.ce_context --pairs-root "$DATA/aug_r1" --out "$DATA/ctx_aug" --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR" \
  --folds ../../artifacts/folds.tsv
for v in r1 r1b ens; do
  case $v in ens) sc="$DATA/aug_ens_scores/scores";; *) sc="$DATA/aug_$v/scores";; esac
  [ -f "$sc/validation.npy" ] || continue
  python -m src.ce_stack --pairs-root "$DATA/aug_r1" --scores-dir "$sc" --context-dir "$DATA/ctx_aug" --extra-dir "$DATA/aug_r1/extra" \
    --test-dir "$TEST_DIR" --out "$DATA/stack_dense_$v"
done
python - <<'PY'
import json
from pathlib import Path
data = Path.home() / "data"
for name in sorted(p.name for p in data.glob("stack_*") if (p / "stack_report.json").exists()):
    r = json.loads((data / name / "stack_report.json").read_text())
    print(f"{name:18s} validation(OOF) {r['validation_oof_macro_f05']:.5f}  holdout {r['holdout']['macro_f05']:.5f}  "
          f"validator {r.get('test', {}).get('validator', {}).get('passed')}")
PY
echo DENSE_ROUTE_DONE
