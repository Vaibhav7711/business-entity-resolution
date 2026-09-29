# Phase 1A candidate-generation pilot

## Scope and method

- Sample: 25,000 S1 entities from validation fold 0, selected by fixed-seed BLAKE2b rank.
- Corpus: all training S2 and S3 records, searched separately within dynamic country partitions.
- Text: Unicode NFKC, case-folding, punctuation and whitespace normalization; separate name and address views.
- Sparse search: hashed character 3–4-gram TF-IDF (262,144 features), corpus-fit IDF per source/country/field, and removal of grams occurring in more than 1% of target records; bounded top-k and no true-link injection.
- Union ranking: reciprocal-rank fusion (constant 60) of exact name, name TF-IDF, and address TF-IDF; unique target IDs retained.
- Recall@k uses the first k candidates per S1 across both target sources. Full candidate recall uses every emitted candidate.
- Every-match percentage includes singleton S1 entities, for which the condition is vacuously true; positive-only rate is also shown.
- Non-ASCII subset: true links where S1 or target name/address contains non-ASCII characters.

## Results

| Route | Link recall | Every match % | Positive-only every match % | R@10 | R@20 | R@50 | R@100 | R@200 | S2 | S3 | Zero candidate % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact_name | 21.844% | 8.32 | 2.88 | 19.347% | 20.067% | 20.970% | 21.519% | 21.784% | 21.718% | 21.963% | 29.04 |
| name_tfidf | 72.834% | 50.08 | 47.11 | 50.331% | 56.697% | 63.713% | 68.680% | 72.834% | 73.233% | 72.458% | 0.01 |
| address_tfidf | 90.113% | 74.74 | 73.24 | 79.887% | 84.153% | 87.172% | 88.622% | 90.113% | 91.691% | 88.625% | 0.00 |
| union | 97.298% | 92.48 | 92.04 | 78.858% | 86.772% | 91.752% | 95.502% | 96.512% | 97.466% | 97.139% | 0.00 |

### Candidate counts and reduction

| Route | Median | p90 | p95 | p99 | Max | Reduction vs country pool |
|---|---:|---:|---:|---:|---:|---:|
| exact_name | 1 | 16 | 61 | 178 | 458 | 99.99981% |
| name_tfidf | 200 | 200 | 200 | 200 | 200 | 99.99627% |
| address_tfidf | 200 | 200 | 200 | 200 | 200 | 99.99627% |
| union | 397 | 399 | 400 | 416 | 698 | 99.99257% |

### Runtime and peak RAM

| Route | Runtime (min) | Peak process RSS (GiB) |
|---|---:|---:|
| exact_name | 3.2 | 1.02 |
| name_tfidf | 7.4 | 1.41 |
| address_tfidf | 16.3 | 1.78 |
| union | 27.6 | 1.96 |

Per-route RAM is the observed process RSS during that stage, including data retained from earlier stages. The union row covers the full pipeline. Exact-name runtime includes the streaming corpus reads shared with the TF-IDF routes.

### Subsets and marginal contribution

| Route | Country recall | Non-ASCII recall (links) | Missing target address recall (links) | Unique extra true links |
|---|---|---:|---:|---:|
| exact_name | India: 15.715% (35,100), US: 26.040% (51,272) | 6.771% (18,034) | 25.966% (3,909) | 7 |
| name_tfidf | India: 64.274% (35,100), US: 78.694% (51,272) | 47.654% (18,034) | 83.372% (3,909) | 4349 |
| address_tfidf | India: 87.134% (35,100), US: 92.152% (51,272) | 88.727% (18,034) | 0.000% (3,909) | 21034 |
| union | India: 95.444% (35,100), US: 98.566% (51,272) | 93.324% (18,034) | 83.500% (3,909) | — |

Unique extra true links means links found by that route and by neither of the other two routes.

## Resources and reproducibility

- Wall time: 27.6 minutes; peak process RAM: 1.96 GiB.
- Config: `configs/phase1a_pilot.json`; seed: 20260925; top-k: 100 per source and sparse route.
- Run locally or in Colab from `code/business_entity_resolution` with `python3 -m pip install -r requirements.txt` and `python3 -m src.evaluate_blocking`.
- sparse-dot-topn 1.2.0 license was checked before installation: Apache-2.0 ([PyPI](https://pypi.org/project/sparse-dot-topn/)).
- No test data was read. No matching model, encoder, reranker, or ensemble was trained.

## Recommendation

Use the exact-name, name TF-IDF, and address TF-IDF union in the next phase, with top-100 per target source for each sparse route. This pilot recovered 97.30% of sampled true links with median 397 candidates per S1. Reassess k and route value on the full validation fold before any matching model work.
