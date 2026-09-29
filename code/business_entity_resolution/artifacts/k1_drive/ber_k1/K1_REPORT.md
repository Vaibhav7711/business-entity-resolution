# K1 — learned first-stage filter (experiment) and full-fold error structure

Frozen B+C+E blocker candidates for all 441,467 fold-0 S1 (Kaggle Phase 1C shards, content-hash verified). Split by S1: train 0–330,999 (out-of-fold filter scores), validation 331,000–385,999, holdout 386,000–441,466.

**Filter decision:** chosen K = 30; K1 gates passed = **False** (retention target 99.800%). Remaining gate: end-to-end matcher F0.5 with vs without filter within 0.001 (measured in K2).

| Measure at chosen K | Value |
|---|---:|
| Validation gold retention | 99.834% |
| Holdout gold retention | 99.843% |
| Transfer (US-trained filter on India validation) | 98.823% |
| Oracle macro F0.5 at K vs no filter (validation) | 0.9919 vs 0.9924 |
| Mean kept candidates per S1 | 30.0 |
| Projected test pairs (with / without filter) | 52M / 944M |

## Retention curve (share of blocker-retrieved gold kept in the top K)

| K | train (OOF) retention | validation retention | holdout retention | US→India transfer | Val oracle F0.5 | Mean kept |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 98.665% | 98.652% | 98.697% | 95.524% | 0.9886 | 10.0 |
| 20 | 99.624% | 99.635% | 99.656% | 98.110% | 0.9913 | 20.0 |
| 30 | 99.826% | 99.834% | 99.843% | 98.823% | 0.9919 | 30.0 |
| 40 | 99.900% | 99.898% | 99.911% | 99.160% | 0.9920 | 40.0 |
| 50 | 99.931% | 99.929% | 99.937% | 99.386% | 0.9922 | 50.0 |
| 60 | 99.953% | 99.949% | 99.957% | 99.503% | 0.9922 | 60.0 |
| 70 | 99.963% | 99.965% | 99.967% | 99.587% | 0.9923 | 70.0 |
| 80 | 99.971% | 99.971% | 99.973% | 99.655% | 0.9923 | 80.0 |
| 100 | 99.982% | 99.984% | 99.985% | 99.762% | 0.9924 | 100.0 |
| 120 | 99.988% | 99.990% | 99.989% | 99.836% | 0.9924 | 120.0 |
| 150 | 99.993% | 99.996% | 99.994% | 99.908% | 0.9924 | 150.0 |
| 200 | 99.998% | 99.999% | 99.998% | 99.962% | 0.9924 | 200.0 |

## Hard negatives: who owns them? (validation + holdout pairs)

Each S2/S3 record belongs to at most one S1. 'Owned by other S1' negatives are resolvable by one-S1-per-record logic when that S1 is also scored.

| Stratum | Positives | Negatives unowned | Negatives owned by another S1 | Share owned | Positive rate |
|---|---:|---:|---:|---:|---:|
| all | 373,103 | 12,577,132 | 47,304,719 | 78.997% | 0.619% |
| same_address | 138,451 | 6,626 | 40,643 | 85.982% | 74.548% |
| strong_name | 215,784 | 1,001,832 | 2,737,066 | 73.205% | 5.456% |
| same_address_strong_name | 78,465 | 1,104 | 3 | 0.271% | 98.609% |
| same_address_weak_name | 23,790 | 3,883 | 39,448 | 91.039% | 35.443% |
| filter_top10 | 368,157 | 157,270 | 579,243 | 78.647% | 33.327% |

## Gold-pair noise patterns (validation + holdout, 373,103 gold pairs)

| Pattern | Share |
|---|---:|
| digits_equal | 61.353% |
| digits_other_mismatch | 20.332% |
| digits_target_subset | 7.081% |
| digits_one_edit_typo | 4.930% |
| target_address_empty | 3.976% |
| digits_equal_after_leading_zero_strip | 3.321% |

Target name scripts: LATIN 349,281, DEVANAGARI 13,473, TELUGU 2,011, KANNADA 1,763, TAMIL 1,729, BENGALI 1,576, GUJARATI 1,540, MALAYALAM 1,025, ORIYA 388, GURMUKHI 317.
Name token-set similarity bins: >=90 271,750, 50-89 66,380, <50 34,973.

Top address word substitutions mined from train gold pairs: street→st (40,653), road→rd (39,743), drive→dr (38,213), avenue→ave (29,470), tx→texas (25,852), maharashtra→mh (18,478), lane→ln (16,135), oh→ohio (14,858), maharashtra→महाराष्ट्र (14,599), il→illinois (13,269), tn→tennessee (11,154), court→ct (9,531), ma→massachusetts (9,471), in→indiana (9,299), az→arizona (8,719), nc→carolina (7,917), nc→north (7,911), md→maryland (7,278), ny→york (7,220), va→virginia (7,054), ny→new (7,027), wa→washington (6,933), ca→california (6,567), karnataka→ka (6,379), uttar→up (6,197).

