# Phase 1B candidate-generation pilot

Same fixed 25,000 S1 validation-fold-0 sample as Phase 1A; all training S2/S3 records remain in the retrieval corpus. No true-link injection or test-set records in retrieval or evaluation.

Subset denominators: 18,034 non-ASCII true links and 3,909 links with a missing target address. Non-ASCII means either S1 or target name/address contains a non-ASCII character.

Preparation note: a broad workspace text search inadvertently displayed a few test TSV lines while locating the problem statement. Those rows were not used to define routes, set thresholds, or compute metrics; all retrieval inputs came from training files.

## Phase 1A reproduction

Baseline union recall: 97.298%; median candidates: 397; candidate reduction vs complete same-country corpus: 99.99257%.
The Phase 1A reproduction comparison is recorded in `reproduction_check.json`.

## One-route additions to the Phase 1A union

| Addition | Link recall | Full positive S1 % | India | US | Non-ASCII | Missing address | Median | p95 | p99 | Max | Reduction | Extra true links | Extra candidates | Route min | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 97.298% | 92.04 | 95.444% | 98.566% | 93.324% | 83.500% | 397 | 400 | 416 | 698 | 99.99257% | 0 | 0 | 0.0 | 1.96 |
| A_name_char_top200 | 97.598% | 92.87 | 95.818% | 98.816% | 93.579% | 86.723% | 597 | 600 | 601 | 742 | 99.98885% | 259 | 4,981,713 | 11.3 | 1.26 |
| B_name_word_tfidf | 97.704% | 93.21 | 96.046% | 98.840% | 93.695% | 87.797% | 530 | 584 | 592.01 | 698 | 99.99023% | 351 | 3,133,284 | 2.5 | 0.94 |
| C_rare_name_token | 97.477% | 92.63 | 95.701% | 98.693% | 93.490% | 85.418% | 399 | 547 | 587 | 698 | 99.99210% | 155 | 634,340 | 0.6 | 0.89 |
| D_address_digit | 97.341% | 92.17 | 95.496% | 98.604% | 93.390% | 83.500% | 398 | 538 | 593 | 774 | 99.99227% | 37 | 409,514 | 0.7 | 0.90 |
| E_legal_suffix_exact | 97.306% | 92.06 | 95.453% | 98.574% | 93.329% | 83.576% | 397 | 400 | 461 | 771 | 99.99256% | 7 | 23,233 | 0.1 | 0.72 |

Each addition is evaluated alone against the reproduced Phase 1A union. Candidate counts are summed over all 25,000 S1 entities. The route runtime is measured across all source/country partitions and excludes the shared corpus read. Peak GiB is observed process RSS during that route (baseline is the complete Phase 1A run).

## Pareto frontier: recall versus mean candidates

| Added routes | Link recall | Mean | Median | p95 | Additional true links | Additional candidates | Accounted total min | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 97.298% | 397.6 | 397 | 400 | 0 | 0 | 27.7 | 1.96 |
| E | 97.306% | 398.6 | 397 | 400 | 7 | 23,233 | 31.1 | 1.96 |
| D | 97.341% | 414.0 | 398 | 538 | 37 | 409,514 | 31.7 | 1.96 |
| D+E | 97.349% | 414.9 | 398 | 543 | 44 | 432,747 | 31.8 | 1.96 |
| C | 97.477% | 423.0 | 399 | 547 | 155 | 634,340 | 31.6 | 1.96 |
| C+E | 97.485% | 423.9 | 399 | 548 | 162 | 657,561 | 31.7 | 1.96 |
| C+D | 97.512% | 439.4 | 400 | 586 | 185 | 1,043,846 | 32.3 | 1.96 |
| C+D+E | 97.520% | 440.3 | 400 | 587 | 192 | 1,067,067 | 32.4 | 1.96 |
| B | 97.704% | 523.0 | 530 | 584 | 351 | 3,133,284 | 33.5 | 1.96 |
| B+E | 97.709% | 523.7 | 531 | 585 | 355 | 3,151,661 | 33.6 | 1.96 |
| B+D | 97.738% | 539.3 | 538 | 664 | 380 | 3,542,784 | 34.2 | 1.96 |
| B+D+E | 97.742% | 540.1 | 538 | 665 | 384 | 3,561,161 | 34.3 | 1.96 |
| B+C | 97.787% | 544.0 | 541 | 677 | 423 | 3,659,688 | 34.1 | 1.96 |
| B+C+E | 97.792% | 544.8 | 541 | 677 | 427 | 3,678,065 | 34.2 | 1.96 |
| B+C+D | 97.819% | 560.4 | 550 | 718 | 450 | 4,069,185 | 34.8 | 1.96 |
| B+C+D+E | 97.823% | 561.1 | 550 | 718 | 454 | 4,087,562 | 34.9 | 1.96 |
| A+B | 97.864% | 702.4 | 707 | 780 | 489 | 7,619,960 | 44.8 | 1.96 |
| A+B+E | 97.865% | 702.8 | 707 | 780 | 490 | 7,628,355 | 44.9 | 1.96 |
| A+B+D | 97.896% | 718.8 | 716 | 842 | 517 | 8,029,449 | 45.5 | 1.96 |
| A+B+D+E | 97.897% | 719.2 | 716 | 842 | 518 | 8,037,844 | 45.6 | 1.96 |
| A+B+C | 97.930% | 720.8 | 718 | 851 | 546 | 8,079,781 | 45.4 | 1.96 |
| A+B+C+E | 97.931% | 721.2 | 718 | 851 | 547 | 8,088,176 | 45.5 | 1.96 |
| A+B+C+D | 97.960% | 737.2 | 727 | 893 | 572 | 8,489,267 | 46.1 | 1.96 |
| A+B+C+D+E | 97.961% | 737.5 | 727 | 893 | 573 | 8,497,662 | 46.3 | 1.96 |

Accounted total runtime adds the measured Phase 1A run, one shared training-corpus scan, each selected route's measured retrieval time, and evaluation time. It is an additive estimate for configurations assembled from the cached route results. Pareto peak RAM is a conservative full-pipeline upper bound. The Phase 1B route-generation and comparison run itself took 21.0 minutes. Full metrics for all 32 route subsets are in `pilot_metrics.json`.

## Route definitions

- A: name character 3–4 gram TF-IDF top-200 per source, using the Phase 1A normalization and corpus IDF.
- B: word unigram/bigram name TF-IDF top-100 per source.
- C: at most 2 rare normalized name tokens, each present in at most 128 targets per source/country; top-100.
- D: at most 2 address digit tokens, each present in at most 128 targets per source/country; top-100.
- E: exact lookup on a name view with up to two trailing legal suffix tokens removed. Only `Corp`, `Corporation`, `Pvt`, `Private`, `Ltd`, and `Limited` from the supplied problem statement are used.

## Recommendation

Select **B + C + E** in addition to the Phase 1A exact-name, name character TF-IDF top-100, and address character TF-IDF top-100 routes. B uses word-name TF-IDF top-100 per source; C uses up to two rare name tokens with a 128-target frequency cap and top-100 per source; E uses exact lookup on the statement-listed suffix-stripped name view.

This configuration retrieved 84,465 of 86,372 links (97.792%), with 93.50% of positive S1 entities fully covered. Median/p95/p99/max candidate counts were 541/677/729/791; candidate reduction was 99.98983%. Relative to Phase 1A, it recovered 427 more true links and introduced 3,678,065 candidates across the pilot.

Country recall was India 96.154% and US 98.914%; non-ASCII recall was 93.767%. Missing-target-address recall rose from 83.500% to 88.949% (+5.45 percentage points). Accounted total runtime was 34.2 minutes and the conservative peak process RSS was 1.96 GiB.

Measured marginal sequence: B added 351 true links and 3,133,284 candidates; C then added 72 links and 526,404 candidates; E then added 4 links and 18,377 candidates, including a small missing-address gain. The E lookup costs about eight seconds across the four target partitions.

Do not retain A: adding it to B+C+E recovered 120 more links but added 4,410,111 candidates, raised the median from 541 to 718, and took 11.3 retrieval minutes. Do not retain D: adding it to B+C+E recovered 27 links, added 409,497 candidates, raised p95 from 677 to 718, and did not improve the missing-address slice.

The 99% link-recall target was not reached by these allowed additions. Even A+B+C+D+E reached 97.961% with median 727 candidates. Use the selected B+C+E configuration for the next full-fold blocking run; this pilot stops here and does not execute that run.
