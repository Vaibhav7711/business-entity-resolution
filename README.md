# Business Entity Resolution at Scale

**Status: complete** (September 2026). Built for the Amazon ML Challenge 2026, business entity resolution task.

Given millions of business records from three noisy sources, find every record in sources 2 and 3 that describes
the same business as each source-1 record. Names come in several scripts, addresses drop or garble numbers, many
businesses share a name or a building, and the test set contains a country (France) that never appears in training.

| | |
|---|---|
| Training data | 2.21M source-1 records, 5.03M + 5.29M target records, 7.64M true links (US, India) |
| Test data | 1,732,544 source-1 records (US, India and France; France is 15% and absent from training) |
| Metric | macro F0.5 per source-1 record, singletons included (precision weighted 2× over recall) |
| Result | macro F0.5 **0.9887** validation / **0.9884** holdout (out of fold); **0.9835** on the hidden test set (public split) |

## Approach

```
S1 ─► blocker: 6 routes (exact / TF-IDF name & address / rare tokens / legal suffix), ~545 candidates per S1
   ─► learned LightGBM filter: top 40 per S1                                          97.8% of true links
   └► dense route: fine-tuned multilingual bi-encoder, + its top 20 not already listed   99.8% of true links
   ─► cross-encoders: multilingual-e5-small (fine-tuned, 2 rounds) on every pair; bge-reranker-v2-m3 on the top 8
   ─► filter stage: top 10 per S1 (99.93% of in-list true links kept)                  candidate set
   ─► LightGBM decision stage, 38 features: model scores, name / address agreement, corpus context,
      consensus with the other strong candidates
   ─► one F0.5-optimal threshold chosen out of fold, one owner per target record        final matches
```

What made the difference, measured on the holdout split:

1. **Dense retrieval** recovers cross-script (Latin vs Indian-script) and variant names the blocker misses: recall
   97.8% → 99.8%, macro F0.5 +0.0065.
2. **Corpus-context features** tell whether a name or address is shared by other records (a name-only match is only
   evidence when the name is unique): +0.004.
3. **List-level features** compare each candidate with the source record's other strong candidates, which are
   independent noisy copies of the same business: +0.0009 in total.

| Stage | Validation | Holdout |
|---|---:|---:|
| String-feature LightGBM, full fold | 0.9601 | 0.9602 |
| Fine-tuned cross-encoder, calibrated threshold | 0.9731 | 0.9721 |
| + corpus-context GBM stacker, one owner (cross-encoder trained on all 331k fold-0 records) | 0.9808 | 0.9808 |
| + dense retrieval route | 0.9871 | 0.9872 |
| + reranker, sibling, address and consensus features (final) | **0.9887** | **0.9884** |
| Ceiling: perfect decisions on the final candidate lists | 0.9992 | 0.9993 |

Tried and rejected on validation evidence: expected-F0.5 set decoding, stacker bagging, a second-stage stacker,
listwise score statistics, a hard street-consistency rule, multilingual-e5-large (undertrained on 2×T4).

The unseen country is the open problem: the model is measurably less certain on France. The final model adds one
rule for it, whose value was tuned with feedback from the public test split, and discloses that.

## Repository

- [`docs/methodology.md`](docs/methodology.md) — the full write-up: data analysis, blocking, models, features,
  results, error analysis.
- [`code/business_entity_resolution/`](code/business_entity_resolution/) — the pipeline: `src/` (all modules),
  `tests/`, `configs/`, small artifacts (fold assignment, filter models, blocking records), Kaggle GPU kernels
  (`kaggle_ce/`) and orchestration (`cloud/`). Its [`README.md`](code/business_entity_resolution/README.md) has the
  exact commands to reproduce both output files end to end, with pinned environments
  (`requirements.txt`, `requirements-gpu.txt`).

The challenge data is not included; `setup_layout.sh` links the official data folder and checks it by sha256.
Tests: from `code/business_entity_resolution`, run `bash setup_layout.sh /path/to/student_resource`, then
`python3 -m pytest -q` (104 tests; CPU only, no downloads).

**Stack:** Python 3.12, LightGBM, scikit-learn, sparse TF-IDF (sparse-dot-topn), RapidFuzz, PyTorch and
Transformers (multilingual-e5-small, MIT; bge-reranker-v2-m3, Apache-2.0), PyArrow. Kaggle 2×T4 GPU kernels for
the neural models, an x86_64 CPU node for blocking, features and the decision stage.

**Team NotAI:** Malav Patel, Vaibhav Jain, Divy Saraswat Saraswat.
