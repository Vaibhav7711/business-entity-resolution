# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** NotAI  
**Team Members:** Malav Patel, Vaibhav Jain, Divy Saraswat Saraswat  
**Submission Date:** 2026-09-29

---

## 1. Executive Summary
Two retrieval routes generate candidates for each Source 1 (S1) record:
- a multi-route blocker with a learned LightGBM filter (top 40);
- a fine-tuned multilingual dense bi-encoder (top 20 more).

Together they contain **99.8%** of the true links. Multilingual cross-encoders then score the candidates: multilingual-e5-small (MIT) in two rounds, plus a fine-tuned BAAI/bge-reranker-v2-m3 (Apache-2.0) on each S1's top 8.

A **filter stage keeps the top 10 candidates per S1**, 5× fewer than the union lists at 99.93% gold retention. That is the set in `candidate_pairs.tsv`. A LightGBM decision stage combines the model scores with corpus-context, sibling-consensus, raw-quirk and address-agreement features. It trains on validation+holdout with out-of-fold threshold selection, then applies a one-owner-per-record rule.

**Results:**
- Out-of-fold macro F0.5 is **0.9887** on validation and **0.9884** on holdout (US/India), against a ceiling of 0.9995 for these candidate lists.
- On the hidden test set (public split), macro F0.5 is **0.98345** for the pooled model and **0.983462** for the final file, which adds one France-only rule (below). The gap comes from France, a country absent from training.

---

## 2. Methodology

### 2.1 Problem Analysis
We analysed the full training data (2.21M S1; 5.03M S2; 5.29M S3; 7.64M gold links) using training folds only.
- **Match structure.** Each S1 has 0 to 6+ matches across S2 and S3; 5.6% are singletons. No S2/S3 record belongs to more than one S1. The number of matches per S1 varies widely, and IDs and row order carry no signal (correlation 0.000).
- **Address numbers.** 64% of true pairs have identical address numbers. The rest show noise:
  - 8.5% have extra unit or flat numbers;
  - 7.6% have dropped numbers;
  - 5.1% drop the house number;
  - 4.8% have one-digit typos;
  - 4.3% have an empty address;
  - 2.9% are landmark-only addresses (India).
- **Names.** 9.4% of true pairs have name token-set similarity below 50: DBA names, reordering, abbreviations. 6.4% of true targets are written in Indian scripts (Devanagari 3.6%, plus Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi) while the S1 name is Latin.
- **The main false-positive pattern is "same address, different business".** 79% of the blocker's wrong candidates are some other S1's true match, and 91% in that same-address stratum.
- **Unseen country.** The test set contains France (15% of test S1), which never appears in training. Every component treats country as an open label.

### 2.2 Solution Strategy
**Approach Type:** two retrieval routes (a multi-route blocker with a learned filter, and a fine-tuned dense bi-encoder), then neural cross-encoders, then a gradient-boosted decision stage with a one-owner rule.
**Core Innovation:** a fine-tuned multilingual cross-encoder trained on hard negatives from our own filter's top ranks. It is applied to every filtered candidate and combined with **corpus-context features** in a gradient-boosted stacker. The context features cover how many S1 and target records share the name or address, empty addresses, exact equality, and duplicates within the list. They answer what a single pair cannot: whether a name-only match is unique. Decisions are resolved globally with one owner per record.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** six routes, run per country label (read from the data, never hard-coded) and per target source, unioned per S1:
  - exact normalised name;
  - name character 3–4-gram TF-IDF (top 100);
  - address character TF-IDF (top 100);
  - name word uni/bigram TF-IDF (top 100);
  - rare name tokens (document frequency ≤ 128);
  - legal-suffix-stripped exact name.
- **Learned filter (K1):** a LightGBM model on cheap features (route evidence plus hashed number, name and address token overlaps with IDF) re-ranks each S1's blocker candidates. Scores for training S1 are out-of-fold. K = 40 was chosen by pre-registered retention gates: validation 99.90%, holdout 99.91%, and 99.16% when a filter trained on US only is applied to India (a proxy for unseen countries).
- **Dense retrieval (second candidate source, §3.1):** a fine-tuned bi-encoder adds its top-20 targets that are missing from the top-40 list.
- **Filter stage (the final candidate set):** the round-1b cross-encoder (§4) scores every pair of the union lists, and the 10 best per S1 by its logit are kept. The decision model runs on exactly these lists, and they are exactly `candidate_pairs.tsv`.
- **Candidate pairs generated (test, 1,732,544 S1, every S1 with candidates):**

  | Stage | Pairs | Per S1 |
  |---|---:|---:|
  | Blocker B+C+E | 943.7M | ~545 |
  | K1 filter, top 40 | 69.3M | 40 |
  | Union with the dense route, as scored by the cross-encoders | 87.5M | ~50 |
  | **Filter stage, top 10 → `candidate_pairs.tsv`** | **17,325,440** | **10** |

- **How true matches were kept:**
  - Every stage was accepted on measured recall, not on cost. The blocker retrieves 97.81% of fold-0 gold links (median 542, p95 679 candidates per S1), the K1 cut was set by the retention gates above, and the dense route lifts the union to **99.83%** of gold links.
  - The top-10 cut keeps **99.93%** of the gold links in the union lists; the oracle macro F0.5 on the final top-10 lists is 0.99921 (validation) and 0.99929 (holdout).
  - Blocking is reproducible across platforms within a documented, gated tolerance (x86 vs ARM TF-IDF near-ties), and all test candidates were produced on x86.

---

### 3.1 Dense retrieval route (second candidate source)
- **Model:** `intfloat/multilingual-e5-small` (MIT) fine-tuned as a bi-encoder on the gold pairs of training folds 1–4 (in-batch negatives, no S1 repeated within a batch). Fold 0, which holds validation and holdout, is never trained on.
- **Retrieval:** each S1 retrieves its top-50 targets by cosine similarity within its own country label. Every dense top-20 target missing from the S1's top-40 list is added.
- **Recall on validation:**
  - The dense route alone reaches 99.2% at k = 10 and 99.7% at k = 20.
  - Top-40 lists plus dense top-20 reach **99.83%** of gold links, against 97.8% for the blocker (India 95.8% and US 82.7% of blocker misses recovered; Indian-script targets 99.7%).
  - The holdout ceiling (oracle macro F0.5 with perfect decisions) rises from 0.992 to **0.9995**.
- **Test:** about 21.7M added pairs.

## 4. Matching Model
**Model type:** fine-tuned multilingual transformer cross-encoders score each (S1, candidate) pair; a LightGBM decision stage combines their scores with name, address and corpus-context features; a global threshold and one owner per S2/S3 record turn probabilities into matches.

### 4.1 Cross-encoders
- **Backbone:** `intfloat/multilingual-e5-small` (MIT, 118M parameters; licence and exact revision checked at load). The model reads "S1 name | address" and "candidate name | address" jointly.
  - **Why this backbone:** a zero-shot check on training folds 1–4 compared name embeddings on true pairs against same-address negatives. e5-small scored AUC 0.972, LaBSE 0.961 and MiniLM 0.949, while string similarity alone scored 0.907 (and fails on cross-script names).
- **Round 1b (scores every pair):**
  - All 331,000 fold-0 training S1, 7.74M pairs: all positives, the 16 hardest negatives by filter rank, and 4 random negatives from the top-40 list.
  - One epoch, binary cross-entropy, AdamW (learning rate 5e-5, linear warmup and decay), fp16, 2×T4 with DDP, word embeddings frozen; 85 minutes at 1.55k pairs/s.
  - Pair AUC is 0.99984 on validation and 0.99983 on holdout. The top-ranked candidate is a true match for 99.94% of S1 that have one.
- **Round 1c (scores the dense-route pairs):** round 1b warm-started for 85 min on the union of the fold-0 top-40 lists and their dense-route additions (24k more, mostly cross-script, gold links). On the dense pairs, holdout AUC rises from 0.99604 to 0.99771.
- **Reranker:** `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M; XLM-R-large with a trained multilingual cross-encoder head), fine-tuned on 2×T4 with 4× gradient accumulation and a learning-rate schedule fitted to the measured throughput. It scores each S1's round-1 top 8 (zero-shot AUC 0.933 on those hard lists).

### 4.2 Decision stage (LightGBM, 38 features per pair)
**Features used:**
- **Name features:** exact normalised name equality (`name_eq`); raw-name edit similarity and equality with case and punctuation kept, since true copies share raw quirks that normalisation erases (`raw_name_ratio_s1`, `raw_name_eq_s1`); fuzzy similarity to the S1's other strongest candidates, which are independent noisy copies of the same entity (`sib_max_sim`, `sib_mean_sim`, `name_sim_s1`, `raw_name_ratio_sib_max`, `raw_name_eq_sib`); how often the name occurs among S1, among targets and within the list (`s1_same_name`, `tg_same_name`, `s1_same_tname`, `list_same_tname`).
- **Address features:** exact equality and empty-address flags (`addr_eq`, `t_addr_empty`, `s1_addr_empty`); digit-free fuzzy token-set, token-sort and partial similarity (`addr_tset`, `addr_tsort`, `addr_partial`); raw address edit similarity (`addr_raw_ratio_s1`); street similarity after generic abbreviation expansion (r→rue, av→avenue, bd→boulevard, rd→road, …) to the S1 and to the sibling candidates (`street_sim_s1_exp`, `street_sim_sib_max`, `street_sim_sib_mean`); first house-number agreement with the siblings (`first_num_agree_sib`); how often the address occurs (`s1_same_addr`, `tg_same_addr`).
- **Other:** the cross-encoder logit (round 1b; round 1c on dense-route rows) and its isotonic-calibrated probability, its rank within the S1's list and gap to the best, and the number of candidates with probability ≥ 0.5 (`logit`, `p`, `logit_rank`, `logit_gap`, `n_p05`); the K1 filter score and rank; the dense bi-encoder cosine, rank and dense-only flag (`dense_score`, `dense_rank`, `dense_only`); the reranker logit, rank and gap (`rr_logit`, `rr_rank`, `rr_gap`).
- Counts are taken within the record's own country label, on the corpus being searched. For training-fold S1, the S1 corpus is subsampled to the test size (1,732,544) so the counts have the same scale on validation and test.

**Model:** LightGBM, binary objective, 600 rounds, learning rate 0.05, 63 leaves, at least 100 rows per leaf, feature fraction 0.9, bagging 0.8, L2 1.0, deterministic (seed 20260927).

**Threshold selection method:** F0.5-optimal, out of fold. The stacker trains on validation + holdout together (110,467 S1) with 4-fold out-of-fold predictions grouped by S1, so no S1 is scored by a model that saw it. The global threshold maximises the sum of the two splits' out-of-fold macro F0.5 on a grid between their individual optima (0.7648). One owner per record is kept because it helps out of fold. The final model is refit on both splits.

**One owner per record:** when several S1 claim the same S2/S3 record, only the highest-scoring claim is kept; at test time every test S1 competes.

**France rule (final submission):** for French S1 only, the top candidate of a list the pooled rule leaves empty is accepted at q ≥ 0.50 (details and evidence in §5).

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** **0.98867** validation and **0.98841** holdout (out-of-fold, US/India); hidden test set (public split) **0.983462** for the final submission.

| Stage | Validation macro F0.5 | Holdout macro F0.5 |
|---|---:|---:|
| String-feature LightGBM baseline (75k S1, 42 features) | — | 0.913 |
| LightGBM, full fold (331k S1, 56 string/number features, top-120 lists) | 0.9601 | 0.9602 |
| Cross-encoder, plain calibrated threshold | 0.97305 | 0.97208 |
| Cross-encoder, stacked logistic policy + one owner | 0.97625 | 0.97514 |
| Cross-encoder + context-feature GBM stacker + one owner | 0.97963 (OOF) | 0.97927 |
| **Round 1b cross-encoder (all 331k fold-0 S1) + context GBM stacker + one owner** | **0.98080 (OOF)** | **0.98079** |
| + dense retrieval route (top-40 ∪ dense top-20), round 1b on all rows, dense features | 0.98712 (OOF) | 0.98724 |
| Pooled stacker (validation+holdout, 4-fold OOF): dense | 0.98750 | 0.98718 |
| Pooled: dense + bge-reranker features | 0.98804 | 0.98780 |
| Pooled: dense + bge-reranker + sibling features | 0.98818 | 0.98796 |
| **Final (G): top-10 filter stage, pooled stacker + sibling-consensus / raw-quirk / address features** | **0.98867** | **0.98841** |
| Oracle on the top-40 candidate lists | 0.99205 | 0.99205 |
| Oracle on the top-40 ∪ dense lists | — | 0.99946 |
| Oracle on the final top-10 lists | 0.99921 | 0.99929 |

Hidden test set (public split): E (dense + round 1c + sibling, pooled) 0.9814; F (+ bge-reranker, address, top-10) 0.9830; **G (+ sibling consensus / raw quirks) 0.98345.**

Measured on the unlabelled test set:
- US and India behave like validation. The class prior re-estimated by EM is ×1.001 of training, and the share of uncertain decisions is the same.
- France has 3–4× more uncertain decisions and far more near-duplicate records per name (17 vs 10 targets share a name).
- Model agreement is lowest on France (91% identical sets vs 97–98%).

France is where the remaining gap between validation and test lies.

**Final France rule (`src/country_rule.py`).** For French S1 only, the top candidate of a list the pooled rule leaves empty is accepted at q >= 0.50 (pooled threshold 0.765); US/India rows are byte-identical to G. France has no labels, so the value comes from feedback on the public test split and is not a held-out estimate: G 0.98345 -> 0.983462 (+804 French matches). A stricter France rule (rank >= 2 needs q >= 0.90) scored lower. On the labelled US/India OOF the same rule is neutral (validation +0.00009, holdout -0.00010), so it is applied to France only.

- The context stacker beats the plain calibrated threshold of the same cross-encoder on holdout by +0.0071 (paired bootstrap 95% CI [+0.0066, +0.0076]).
- On holdout: mean precision 0.996, mean recall 0.953, singleton accuracy 0.990.
  - The remaining loss is mostly recall.
  - 2.2% of gold links are never retrieved by the blocker. The largest group is Latin S1 names whose target is written in an Indian script.
  - The rest are in-list matches the threshold rejects: empty-address targets, heavy abbreviations, and DBA names.
- **Common false positives:** another business at the same address, especially franchises and shared buildings; duplicate claims across S1 (reduced by the one-owner rule).
- **Common false negatives:** gold not retrieved by blocking (2.2% of links; the largest single category in the LightGBM error budget, worth +0.009 macro F0.5 if fixed); heavily abbreviated or DBA names; Latin versus Indian-script names with dropped address numbers.
- **Tried and rejected on validation:**
  - a per-S1 expected-F0.5 set decision (holdout 0.97436 against 0.97507);
  - a gradient-boosted stacker without context features (+0.0004, within noise);
  - "sibling" features, meaning the best logit among other candidates that share the name or address (+0.0002);
  - averaging round 1 and round 1b (0.98064 OOF against 0.98080), or stacking them side by side (0.98090 against 0.98080; holdout 0.98080 against 0.98079, i.e. no gain);
  - multilingual-e5-large on 2×T4: undertrained within the time budget (cascade AUC 0.99802 against 0.99855 for e5-small).
  - "Competition" features: each target's best dense score among other S1 in a corpus sample. They gave +0.0003 but were excluded, because the corpora differ in shape (4.7 against 5.8 targets per S1). The features therefore shift between validation and test (82% against 77% of candidates listed by another S1).
  - S2/S3 source-aware list features (0.98069 against 0.98081).
  - Stacker hyperparameters (four settings; the default was best).
- **Remaining error (holdout, dense stack):**
  - 70% of missed links are **name-only targets** (empty address). Recall on them is 0.54, against 0.991 for targets with an address.
  - The stacker is well calibrated on them: probability 0.5–0.6 means 49% are true. Given only name and country, many of these are genuinely ambiguous. The data contains same-name and near-name distractor records.

---

## 6. Conclusion
Recall comes from two complementary retrieval routes, a multi-route blocker and a fine-tuned dense bi-encoder. They raise the reachable ceiling from 0.992 to 0.9995. Precision comes from multilingual cross-encoders and a list-aware LightGBM decision stage.

The measured gains, in order of size:
1. **Dense retrieval**, recovering cross-script and variant names: +0.0065 holdout.
2. **Corpus-context features**, telling whether a name or address is shared: +0.004.
3. **List-level features**, comparing a candidate with the S1's other strong candidates:
   - reranker rank and gap: +0.0006;
   - sibling consensus and raw-quirk agreement: +0.0003, confirmed on the test set.

Several ideas failed their holdout gates: expected-F decoding, stacker bagging, a second-stage stacker, and a hard street-consistency rule. Deciding by holdout evidence kept all of them out of the submission.

The main open problem is generalisation to an unseen country. French distractors reuse the name and street number on another street, and true duplicates swap region for department. The model is systematically less certain there.

---

## 7. Other Relevant Information
- **Data and fair play:** only the supplied training and test files are used; no external lookup, APIs, geocoding or enrichment. Test records were never used for training or threshold selection. Unlabelled test statistics were used only as diagnostics (class prior by EM, agreement between models, feature shift by country). The one choice informed by test feedback is the value of the France rule, taken from public test-split feedback and disclosed in §5.
- **Pretrained models:** `intfloat/multilingual-e5-small` (MIT, 118M) and `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M), both far below 8B parameters. The code refuses any other licence and records the exact revision it loaded.
- **Country as an open set:** the pipeline reads the country label from the data and never hard-codes US or India; France (15% of test S1, absent from training) runs through the same pipeline. The France rule is a generic per-label rule (`--country`), not a code path for one country.
- **Compute:** one x86_64 CPU node (8 vCPU, 61 GB RAM) for blocking, features and the decision stage; Kaggle 2×T4 GPU kernels for the cross-encoders, the bi-encoder and the reranker, driven through the Kaggle API (`cloud/kaggle_ce.sh`).
- **Reproducibility:** exact commands in `code/business_entity_resolution/README.md`, pinned environments, deterministic LightGBM, and the sha256 sums of both output files. From the saved stacker, `src/country_rule.py` reproduces the submitted `matching_results.tsv` byte for byte.
- **Evaluation discipline:** fold 0 is split by position into cross-encoder training, validation and holdout. Every reported score is out of fold or held out. Components were adopted only through gates stated before the run, and the ideas that failed them are listed in §5.

---

## Appendix
### A. Code Artefacts
`code/business_entity_resolution/` holds the complete pipeline: all source in `src/`, `README.md` with the exact commands, `requirements.txt` and `requirements-gpu.txt` with pinned versions, `configs/`, the small artifacts later steps read (`artifacts/`: fold assignment, K1 filter models, blocking gate records) and the tests.
- **Entry points:** `setup_layout.sh` (layout and data check) → README steps 0–10 (folds, blocking, K1 filter, top-40 lists, GPU kernels through `cloud/kaggle_ce.sh`, dense route, augmented lists) → `run_final_stages.sh` (steps 11–16: features, top-10 filter stage, pooled decision stage, France rule, official validator, checksums) → `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- **Main modules:**
  - blocking: `src/evaluate_phase1c.py`, `src/block_test.py`, `src/blocking.py`, `src/phase1b_routes.py`;
  - K1 filter and top-40 lists: `src/k1_filter.py`, `src/topk_export.py`;
  - cross-encoders and reranker: `src/ce_model.py`, `kaggle_ce/ce_kernel.py`; cascade: `src/ce_cascade.py`;
  - dense route: `src/bi_encoder.py`, `kaggle_ce/bi_kernel.py`, `src/dense_merge.py`;
  - features: `src/ce_context.py` (corpus context), `src/ce_join.py` (reranker scores), `src/sib_context.py`, `src/addr_context.py`, `src/sib2_context.py`;
  - filter stage: `src/restrict_topk.py`; decision stage and writers: `src/ce_stack.py`, `src/ce_policy.py`; France rule: `src/country_rule.py`;
  - orchestration: `cloud/kaggle_ce.sh`, `cloud/acc2_r1c.sh`, `cloud/rr_test12.sh`, `cloud/dense_pipeline2.sh`, `cloud/finish_dense_c2.sh`, `cloud/after_rr2.sh`, with checks in `cloud/check_scores.py`.
- **Tests:** `python3 -m pytest -q`.

### B. Additional Results
- **Filter retention by K** (validation / holdout / US→India transfer):
  - K = 30: 99.83% / 99.84% / 98.82%
  - K = 40: 99.90% / 99.91% / 99.16%
  - K = 120: 99.99% / 99.99% / 99.84%
- **Cross-encoder throughput on 2×T4:** training 1.6k pairs/s; scoring 6.4k pairs/s.
