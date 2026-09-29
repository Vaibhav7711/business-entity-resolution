# Business Entity Resolution — ML Challenge 2026 (team NotAI)

For each test Source 1 (S1) record, find the matching Source 2/3 records. The pipeline writes both submission
files: `matching_results.tsv` (the final matches, the scored file) and `candidate_pairs.tsv` (the exact
candidate lists the final decision model runs on).

## Pipeline

```
S1 ─► blocker B+C+E (6 routes, ~545 candidates/S1) ─► K1 LightGBM filter ─► top-40 per S1         (97.8% of gold)
   └► dense route: e5-small bi-encoder (trained on folds 1-4 gold) ─► + dense top-20 not in top-40  (99.8% of gold)
   ─► cross-encoders: e5-small round 1b on every pair (dense pairs rescored by round 1c, warm-started on the
      union pairs); bge-reranker-v2-m3 (fine-tuned) on each S1's round-1 top 8
   ─► filter stage: top 10 per S1 by the round-1b logit (99.93% of in-list gold kept)       [candidate_pairs.tsv]
   ─► pooled LightGBM decision stage over cross-encoder, reranker and dense scores, corpus-context, sibling,
      address, sibling-consensus and raw-quirk features; out-of-fold threshold; one owner per S2/S3 record
   ─► France rule: for French S1 only, the top candidate of an otherwise empty list at q >= 0.50  [matching_results.tsv]
```

Final model: out-of-fold macro F0.5 0.98867 (validation) and 0.98841 (holdout), both US/India; hidden test set
(public split) 0.983462. Test: 1,732,544 S1, 17,325,440 candidate pairs, 5,810,718 matches (94.16% of S1 non-empty).

Only the supplied data is used. Pretrained weights: `intfloat/multilingual-e5-small` (MIT, 118M parameters) and
`BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M); licence and revision are checked when they load
(`src/ce_model.py`, pinned revisions in `requirements-gpu.txt`).

## Reproduce end to end

Hardware used: CPU steps on an x86_64 Linux node (8 vCPU, 61 GB RAM; blocking must run on x86_64, see
`configs/phase1c_fold0.json` -> platform tolerance); GPU steps as Kaggle kernels on 2x Tesla T4, driven through
the Kaggle API by `cloud/kaggle_ce.sh` (`kaggle_ce/ce_kernel.py`, `kaggle_ce/bi_kernel.py`). Every command runs
from this folder, `code/business_entity_resolution`. Every step is resumable or cheap to rerun.

### 0. Layout, environment, data

```bash
bash setup_layout.sh /path/to/student_resource   # official data: checks sha256, links it; copies configs/, artifacts/ to ../../
bash cloud/setup_node.sh                         # ~/venv: Python 3.12 + requirements.txt (pinned); runs the tests
. ~/venv/bin/activate
export D=$HOME/data TR=../../student_resource/dataset/train TE=../../student_resource/dataset/test
P='(scores/[a-z]+\.npy|[a-z_]+\.json|[a-z_]+\.log)$'          # kernel outputs to download
```

GPU steps only: a Kaggle account with phone verification (GPU + internet), its API token in `~/.kaggle_api_token`.
Our scripts name our account; switch them to yours:

```bash
export KAGGLE_OWNER=<your-kaggle-username>; U=$KAGGLE_OWNER
sed -i "s/vaibhav0383/$U/g" cloud/*.sh
wait_kernel() { until kaggle kernels status $U/$1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 120; done; kaggle kernels status $U/$1; }
```

### 1. Folds

`artifacts/folds.tsv` ships with this folder (5 folds by S1 hash; fold 0 = validation fold). To regenerate it:

```bash
python3 src/make_folds.py --ground-truth $TR/train_ground_truth.tsv --source1 $TR/train_source1.tsv \
  --output ../../artifacts/folds.tsv --summary ../../artifacts/folds_summary.json --folds 5 --reused-target-count 0
```

### 2. Blocking: fold 0 and test

```bash
SKIP_PILOT=1 WORK_DIR=../../artifacts/phase1c/work bash run_phase1c_parallel.sh     # fold 0 -> artifacts/phase1c/
for r in 0-23 24-46 47-69; do SHARDS=$r bash run_test_blocking_kaggle.sh; done      # test -> artifacts/test_blocking/
```

We ran both on Kaggle CPU notebooks (x86_64) with these scripts; `artifacts/phase1c/` holds that run's gate record,
metrics and manifest. `SKIP_PILOT=1` uses the recorded pilot gate; without it the script first repeats the 25k pilot,
which needs the Phase 1A/1B reference candidates (development history below).

### 3. K1 filter and top-40 lists

```bash
python3 -m src.k1_filter --config ../../configs/k1_filter.json --phase1c-work-dir ../../artifacts/phase1c/work \
  --work-dir $D/k1_work --output-dir ../../artifacts/k1_drive/ber_k1 --stage all      # models ship in artifacts/k1_drive
PHASE2A_THREADS=1 python3 -m src.topk_export --split fold0 --config ../../configs/k2_matcher.json \
  --k1-dir ../../artifacts/k1_drive/ber_k1 --phase1c-work-dir ../../artifacts/phase1c/work \
  --work-dir $D/topk_fold0_work --output-dir $D/pairs_fold0 --keep-k 40 --workers 8
PHASE2A_THREADS=1 python3 -m src.topk_export --split test --config ../../configs/k3_inference.json \
  --k1-dir ../../artifacts/k1_drive/ber_k1 --work-dir $D/topk_test_work --output-dir $D/pairs_test --keep-k 40 \
  --workers 8 --test-dir $TE --blocking-root ../../artifacts/test_blocking
mkdir -p $D/pairs_ce1 && ln -sfn $D/pairs_fold0/validation $D/pairs_ce1/validation \
  && ln -sfn $D/pairs_fold0/holdout $D/pairs_ce1/holdout && ln -sfn $D/pairs_test/test $D/pairs_ce1/test
```

Fold 0 splits by position: train (331,000 S1; cross-encoder training), validation (55,000), holdout (55,467).

### 4. Round-1 and round-1b cross-encoders (GPU)

```bash
bash cloud/kaggle_ce.sh code; bash cloud/kaggle_ce.sh fold0; bash cloud/kaggle_ce.sh test   # private datasets
bash cloud/kaggle_ce.sh ready                                   # repeat until all three are "ready"
bash cloud/kaggle_ce.sh push                                    # ber-ce: round 1, configs/ce.json
bash cloud/kaggle_ce.sh push ber-ce-r1b ce_r1b.json all - $U/ber-ce-code,$U/ber-ce-pairs-fold0,$U/ber-ce-pairs-test -
wait_kernel ber-ce
bash cloud/kaggle_ce.sh push ber-ce-test ce.json score test $U/ber-ce-code,$U/ber-ce-pairs-test $U/ber-ce
wait_kernel ber-ce-test; wait_kernel ber-ce-r1b
kaggle kernels output $U/ber-ce -p $D/ce_out --file-pattern "$P" -o
kaggle kernels output $U/ber-ce-test -p $D/ce_test_out --file-pattern '(scores/test\.npy|test_k\.json|score_log\.json)$' -o
T=$(find $D/ce_test_out -name test.npy -path '*scores*' | head -1)
cp $T $D/ce_out/ce/scores/test.npy && cp "$(dirname "$(dirname $T)")/test_k.json" $D/ce_out/ce/test_k.json
kaggle kernels output $U/ber-ce-r1b -p $D/ce_r1b --file-pattern "$P" -o
R1B=$(dirname "$(find $D/ce_r1b -name validation.npy -path '*scores*' | head -1)")
```

### 5. Round-1 cascade (top 8 per S1, the reranker's input)

```bash
python -m src.ce_cascade --pairs-root $D/pairs_ce1 --scores-dir $D/ce_out/ce/scores --out $D/pairs_cascade --n 8
```

### 6. Dense route (GPU): bi-encoder lists, new pairs, round-1b scores

```bash
KERNEL=bi_kernel.py bash cloud/kaggle_ce.sh push ber-bi bi.json all - "$U/ber-ce-code,$U/ber-ce-pairs-fold0"
wait_kernel ber-bi
kaggle kernels output $U/ber-bi -p $D/bi_out --file-pattern "(dense/.*|[a-z_]+\.json|[a-z_]+\.log)$" -o
DENSE=$(dirname "$(find $D/bi_out -path '*dense/validation' -type d | head -1)")
python -m src.dense_merge --stage new --base-root $D/pairs_ce1 --dense-root $DENSE --new-root $D/dense_new \
  --train-dir $TR --max-rank 20
S=$D/kaggle_stage/dense_new; rm -rf $S && mkdir -p $S/notes
for s in validation holdout test; do cp -al $D/dense_new/$s $S/$s; done
echo "Dense-route pairs not in the top-40 lists (dense rank < 20)." > $S/notes/README.txt
printf '{"title": "ber-dense-new", "id": "%s/ber-dense-new", "licenses": [{"name": "other"}]}' $U > $S/dataset-metadata.json
kaggle datasets create -p $S --dir-mode zip -q                  # wait until its files are listed
bash cloud/kaggle_ce.sh push ber-dense-score ce_r1b.json score validation,holdout,test "$U/ber-ce-code,$U/ber-dense-new" "$U/ber-ce-r1b"
wait_kernel ber-dense-score
kaggle kernels output $U/ber-dense-score -p $D/dense_scores --file-pattern "$P" -o
NEWS=$(dirname "$(find $D/dense_scores -name validation.npy -path '*scores*' | head -1)")
python cloud/check_scores.py --scores $NEWS --rows-json $D/dense_new/dense_merge_new.json --splits validation holdout test
```

### 7. Round 1c (GPU): round 1b warm-started on the union training pairs, scoring the dense pairs

```bash
kaggle kernels output $U/ber-bi -p $D/bi_model_dl --file-pattern "(model/.*|train_log\.json)$" -o
kaggle kernels output $U/ber-ce-r1b -p $D/r1b_model_dl --file-pattern "(model/.*|train_log\.json)$" -o
cp ~/.kaggle_api_token ~/.kaggle_api_token2 && echo $U > ~/.kaggle_user2   # we ran it on a second account; one works
bash cloud/acc2_r1c.sh        # dense lists for fold-0 train S1 -> union pairs -> ber-r1c -> 6 test shards -> $D/r1c_new/scores
```

### 8. Reranker (GPU): bge-reranker-v2-m3 on the round-1 top 8

```bash
S=$D/kaggle_stage/rr_pairs; rm -rf $S && mkdir -p $S/notes && cp -al $D/pairs_fold0/train $S/train
for s in validation holdout test; do cp -al $D/pairs_cascade/$s $S/$s; done
echo "Reranker pairs: train/ = fold-0 top-40 training pairs; validation/, holdout/, test/ = round-1 top 8." > $S/notes/README.txt
printf '{"title": "ber-rr-pairs", "id": "%s/ber-rr-pairs", "licenses": [{"name": "other"}]}' $U > $S/dataset-metadata.json
kaggle datasets create -p $S --dir-mode zip -q                  # wait until its files are listed
bash cloud/kaggle_ce.sh push ber-rr-train ce_rerank_t4.json all validation,holdout "$U/ber-ce-code,$U/ber-rr-pairs"
wait_kernel ber-rr-train
kaggle kernels output $U/ber-rr-train -p $D/rr_train --file-pattern "$P" -o
OUT=$(dirname "$(find $D/rr_train -name train_log.json | head -1)")
mkdir -p $D/rerank/scores && cp $OUT/scores/validation.npy $OUT/scores/holdout.npy $D/rerank/scores/
for part in a:0,1,2 b:3,4,5 c:6,7,8 d:9,10,11; do                # test: 12 shards on 4 kernels
  KERNEL_ENV="{\"CE_TEST_SHARDS\": \"12\", \"CE_TEST_ONLY\": \"${part#*:}\"}" bash cloud/kaggle_ce.sh push \
    ber-rr-t12-${part%%:*} ce_rerank_t4.json score test "$U/ber-ce-code,$U/ber-rr-pairs" "$U/ber-rr-train"
done
for k in a b c d; do
  wait_kernel ber-rr-t12-$k
  kaggle kernels output $U/ber-rr-t12-$k -p $D/rr12_$k --file-pattern "(shard.*\.npy|[a-z_]+\.json|[a-z_]+\.log)$" -o
  find $D/rr12_$k -name "test.shard*of12.npy" -exec cp {} $D/rerank/scores/ \;
done
python -c "from pathlib import Path; from src.ce_model import merge_shards; import sys; sys.exit(0 if merge_shards(Path('$D/rerank/scores'), 'test', 12) else 1)"
python cloud/check_scores.py --scores $D/rerank/scores --rows-json $D/pairs_cascade/cascade_summary.json --splits validation holdout test
```

(Our run: `cloud/rr_test12.sh`, two kernels on each of two accounts.)

### 9. Augmented lists and corpus-context features

```bash
AUG="--base-root $D/pairs_ce1 --dense-root $DENSE --new-root $D/dense_new --base-scores $R1B"
python -m src.dense_merge --stage augment $AUG --new-scores $NEWS --aug-root $D/pairs_aug
python -m src.dense_merge --stage augment $AUG --new-scores $D/r1c_new/scores --aug-root $D/pairs_aug_c
python -m src.ce_context --pairs-root $D/pairs_aug --out $D/ctx_aug --train-dir $TR --test-dir $TE --folds ../../artifacts/folds.tsv
```

### 10. Reranker features on the augmented lists

```bash
python -m src.ce_join --src-root $D/pairs_cascade --src-scores $D/rerank/scores --dst-root $D/pairs_aug --out $D/rr_on_aug --name rr
```

### 11-16. Features, filter stage, decision stage, France rule, checks (CPU)

```bash
bash run_final_stages.sh
```

It runs, in order: `sib_context` and `addr_context` on the augmented lists (11); `restrict_topk`, the top 10 per S1
by the round-1b logit carrying every score and feature file (12); `sib2_context` on the top-10 lists (13);
`ce_stack --pooled` (14, the decision stage: out-of-fold threshold 0.7648 and one owner, written to
`$D/final_g/{stack_report.json, stacker.txt, calibrator.json, output/}`); `country_rule --empty-top 0.50` (15,
France only: `$D/final_country/output/`); the official validator and the sha256 sums (16). Expected sums of the
submitted files:

```
1cc9ef3b81fb224e15e602b81bc07a4790aeb59b022ad0a9b04f835572587e0c  matching_results.tsv
ae661ef213d636b37aed152b54c18d434e7884f1b97a28e3c09a79d64e61b9fb  candidate_pairs.tsv
```

`src/country_rule.py` reproduces these byte for byte from the saved stacker. The France rule's value was chosen with
feedback from the public test split (France has no labels), so it is not a held-out estimate; on the labelled US/India
out-of-fold predictions the same rule is neutral (+0.00009 validation, -0.00010 holdout).

## Tests

`python3 -m pytest -q` (CPU; the model tests build tiny local models and download nothing).

---

## Development history (earlier phases)

Working notes from the earlier phases, kept for provenance. Some paths below are relative to where they
were run at the time; the reproduction section above is the authoritative, current set of commands.


Phase 0 utilities and Phase 1A/1B candidate-generation pilots for the ML Challenge 2026 business entity resolution task.

## Phase 0 audit

```bash
python3 src/audit_data.py \
  --dataset-dir ../../../student_resource/dataset \
  --output-dir ../../../artifacts/audit

python3 src/make_folds.py \
  --ground-truth ../../../student_resource/dataset/train/train_ground_truth.tsv \
  --source1 ../../../student_resource/dataset/train/train_source1.tsv \
  --output ../../../artifacts/folds.tsv \
  --summary ../../../artifacts/folds_summary.json \
  --folds 5 \
  --reused-target-count 0
```

The fold command is deliberately guarded. If the audit finds an S2/S3 target
linked to more than one S1 entity, build connected components instead of using
independent S1 hashing.

## Local scoring

```bash
python3 src/score.py \
  --truth path/to/ground_truth.tsv \
  --prediction path/to/predictions.tsv
```

## Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m pytest -q tests/test_phase1a.py
```

## Candidate-generation pilot

The paths below assume this directory is copied with its `student_resource`,
`artifacts`, and `configs` sibling tree intact. In Colab, mount/copy the same
repository tree first, then run:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.evaluate_blocking
python3 -m src.check_phase1a_reproduction
python3 -m src.evaluate_phase1b
```

The pilot selects exactly 25,000 validation-fold-0 S1 entities and searches
the entire **training** S2/S3 corpus by dynamic country. It does not read test
data or train a matching model. The configuration is at
`../../configs/phase1a_pilot.json`; output goes to `../../artifacts/phase1a`.
Phase 1B uses `../../configs/phase1b_pilot.json` and writes
`../../artifacts/phase1b`. It evaluates the five specified route changes on
the same S1 IDs and stops after recommending one candidate configuration.

## Phase 1C full-fold blocker evaluation

Frozen B+C+E blocker on all 441,467 fold-0 S1 records, checkpointed per
target source x dynamic country x route x 25k S1 shard. Shard 0 is exactly the
Phase 1A/1B pilot sample. `--scope` is mandatory, and `full` is refused until
`pilot_reproduction.json` (exact) and `benchmark_eta.json` exist.

```bash
python3 -m src.evaluate_phase1c --config ../../configs/phase1c_fold0.json --scope pilot
python3 -m src.evaluate_phase1c --config ../../configs/phase1c_fold0.json --scope benchmark
python3 -m src.evaluate_phase1c --config ../../configs/phase1c_fold0.json --scope full
bash run_phase1c_parallel.sh   # cloud CPU: parallel shard workers + final evaluation
```

Resume by rerunning the same command. `--stop-after-tasks N` exits cleanly for
resume tests. `--only-source/--only-country/--only-route` generate disjoint
shards for parallel workers without evaluating. See
`../../artifacts/phase1c/CLOUD_RUNBOOK.md`.
