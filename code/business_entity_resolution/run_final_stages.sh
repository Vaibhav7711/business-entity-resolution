#!/usr/bin/env bash
# README steps 11-16 in one go (CPU): from the augmented candidate lists, their context features and the reranker
# features to output/{matching_results,candidate_pairs}.tsv, then the official validator and the sha256 sums.
#
# Needs in $D (default ~/data), from README steps 9-10: pairs_aug/, pairs_aug_c/, ctx_aug/, rr_on_aug/.
# FRANCE_RULE holds the step-16 flags of src/country_rule.py (default: the final submission's rule).
# Usage (from code/business_entity_resolution): bash run_final_stages.sh
set -euo pipefail
D=${D:-$HOME/data}; TR=../../student_resource/dataset/train; TE=../../student_resource/dataset/test
FRANCE_RULE=${FRANCE_RULE:---empty-top 0.50}
for x in pairs_aug pairs_aug_c ctx_aug rr_on_aug; do [ -e "$D/$x" ] || { echo "missing $D/$x (README steps 9-10)"; exit 1; }; done

# 11. Sibling-name and address-agreement features on the augmented lists.
python -m src.sib_context --pairs-root $D/pairs_aug --scores-dir $D/pairs_aug/scores --train-dir $TR --test-dir $TE --out $D/sib_aug
python -m src.addr_context --pairs-root $D/pairs_aug --train-dir $TR --test-dir $TE --out $D/addr_aug
# 12. Filter stage: top 10 per S1 by the round-1b logit -> the lists in candidate_pairs.tsv, with every feature file.
python -m src.restrict_topk --pairs-root $D/pairs_aug_c --rank-scores $D/pairs_aug/scores --k 10 --out $D/top10 \
  --scores-dirs $D/pairs_aug_c/scores:$D/top10/scores \
  --feature-dirs $D/ctx_aug:$D/top10_ctx $D/pairs_aug_c/extra:$D/top10_extra $D/rr_on_aug:$D/top10_rr \
                 $D/sib_aug:$D/top10_sib $D/addr_aug:$D/top10_addr
# 13. Sibling-consensus and raw-quirk features on the top-10 lists.
python -m src.sib2_context --pairs-root $D/top10 --scores-dir $D/top10/scores --train-dir $TR --test-dir $TE --out $D/top10_sib2
# 14. Pooled decision stage (validation + holdout, 4-fold out-of-fold threshold, one owner) and the test output.
FEATURES="--pairs-root $D/top10 --scores-dir $D/top10/scores --context-dir $D/top10_ctx \
  --extra-dir $D/top10_extra $D/top10_rr $D/top10_sib $D/top10_addr $D/top10_sib2"
python -m src.ce_stack --pooled $FEATURES --test-dir $TE --out $D/final_g
# 15. France rule on top of the saved stacker (US and India rows unchanged).
python -m src.country_rule $FEATURES --stack-dir $D/final_g --test-dir $TE --out $D/final_country $FRANCE_RULE
# 16. Official validator and checksums.
python ../../student_resource/utils/validate_submission.py --matching $D/final_country/output/matching_results.tsv \
  --candidate $D/final_country/output/candidate_pairs.tsv --test-dir $TE --check-ids | tail -1
(cd $D/final_country/output && { command -v sha256sum >/dev/null && sha256sum *.tsv || shasum -a 256 *.tsv; })
