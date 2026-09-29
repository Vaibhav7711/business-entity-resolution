# Colab pilot reproduction — cross-platform diagnosis

The pilot scope was run on Google Colab on a fresh work directory:
- Intel Xeon @ 2.20 GHz, 2 vCPU, x86_64;
- Python 3.13.15, Unicode 15.1.0;
- numpy 2.4.2, scipy 1.17.1, scikit-learn 1.8.0, sparse-dot-topn 1.2.0.

The training inputs were SHA-256-verified identical to the local files.

Result: `all_equal = false`. The full run was blocked by the gate as designed.

## Per-route shard 0 comparison with the Apple M1 run (Python 3.12)

| Route | S2/India | S2/US | S3/India | S3/US |
|---|---|---|---|---|
| exact_name | identical | identical | identical | identical |
| rare_name | identical | identical | identical | identical |
| suffix_exact | identical | identical | identical | identical |
| name_char | IDs differ | IDs differ | IDs differ | IDs differ |
| address_char | IDs differ | IDs differ | IDs differ | IDs differ |
| name_word | IDs differ | IDs differ | IDs differ | IDs differ |

Every task has an identical candidate count on both platforms.

## Pilot metric effect (Colab vs Phase 1B / M1)

| Metric | Phase 1B / M1 | Colab |
|---|---:|---:|
| Retrieved true links | 84,465 | 84,464 |
| Link recall | 97.79211% | 97.79095% |
| Positive every-match | 93.500% | 93.496% |
| US recall | 98.9136% | 98.9117% |
| Candidate total | 13,618,996 | 13,618,606 |
| Phase 1A union candidate total | 9,940,931 | 9,940,919 |

The following are identical on both platforms:
- India recall;
- non-ASCII recall;
- missing-address recall;
- median, p95, p99 and max candidate counts.

## Interpretation

The routes with no floating-point computation (exact, rare-token, suffix) are bit-identical. This rules out drift in data, normalization or Unicode. The TF-IDF routes keep full top-k rows but swap near-tied IDs at the top-k boundary. This is consistent with x86 and ARM rounding differences in the sparse dot products.

It is platform nondeterminism, not implementation drift. Its effect is 1 true link of 86,372 on the pilot, well inside the 0.5 pp recall gate. However, it does not meet the strict exact-reproduction criterion.

## Decision

The candidate IDs of the frozen blocker depend on the CPU architecture. Therefore:
- one platform must produce the validation candidates, Phase 2 training pairs, and final test candidates;
- the Apple M1 reproduces Phase 1B exactly and is the reference platform;
- the Colab shards are not mixed with the M1 shards.
