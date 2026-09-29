#!/usr/bin/env bash
# Round-2 check (run on the node, from code/business_entity_resolution): once e5-large training is done, score the
# validation/holdout cascade (top-N per S1 by round 1) and tune the stacked policy (round-2 logit + round-1 logit/rank).
set -uo pipefail
. "$HOME/venv/bin/activate"
export KAGGLE_API_TOKEN=$(cat "$HOME/.kaggle_api_token")
DATA=${DATA:-$HOME/data}
PATTERN='(scores/[a-z]+\.npy|[a-z_]+\.json|[a-z_]+\.log)$'
wait_kernel() {
  while true; do
    s=$(kaggle kernels status "vaibhav0383/$1" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z_]+")
    echo "$(date +%H:%M) $1: $s"
    case "$s" in *COMPLETE) return 0;; *ERROR|*CANCEL*) return 1;; esac
    sleep 120
  done
}
wait_kernel ber-ce-large-train || { echo "round-2 training failed"; exit 1; }
kaggle kernels output vaibhav0383/ber-ce-large-train -p "$DATA/ce2_train" --file-pattern '(train_log\.json|ce_run\.log)$' -o > /dev/null
head -c 900 "$(find "$DATA/ce2_train" -name train_log.json | head -1)"; echo
bash cloud/kaggle_ce.sh push ber-ce-large-val ce_large.json score validation,holdout \
  vaibhav0383/ber-ce-code,vaibhav0383/ber-ce-pairs-cascade-val vaibhav0383/ber-ce-large-train
sleep 120
wait_kernel ber-ce-large-val || { echo "round-2 validation scoring failed"; exit 1; }
kaggle kernels output vaibhav0383/ber-ce-large-val -p "$DATA/ce2_val" --file-pattern "$PATTERN" -o > /dev/null
S2=$(dirname "$(find "$DATA/ce2_val" -name validation.npy -path '*scores*' | head -1)")
cat "$(dirname "$S2")/score_log.json"; cat "$(dirname "$S2")/eval.json"
python -m src.ce_policy --pairs-root "$DATA/pairs_cascade_val" --scores-dir "$S2" --out "$DATA/policy_r2_val" --stage tune \
  --train-dir ../../student_resource/dataset/train
head -c 2500 "$DATA/policy_r2_val/POLICY_REPORT.md"
echo "ROUND2 CHECK DONE"
