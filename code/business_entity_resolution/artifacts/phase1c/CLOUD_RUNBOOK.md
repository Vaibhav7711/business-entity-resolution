# Phase 1C — cloud CPU runbook for the full fold-0 blocker run

The algorithm, config, code, and pinned environment are identical to the local runs. The cloud machine only adds parallel shard workers. Worker count and thread count do not change candidate IDs or scores. This is covered by unit tests and a real-data check at 1/4/8 threads.

## Status before the cloud run (local Apple M1, 8 GiB RAM)

| Evidence | Result | File |
|---|---|---|
| Pilot reproduction (shard 0 = exact 25k Phase 1B IDs, same order) | 35/35 metric checks exactly equal. 0 per-S1 candidate mismatches in all 6 routes. Dense scores bit-identical. | `local_m1/pilot_reproduction.json` |
| Resume | Stopped intentionally after 8 tasks. The resumed invocation skipped the 8 completed tasks and the S2/India scan. | `local_m1/work_invocations.jsonl` |
| Benchmark (shards 0–2 = 75k; shards 1–2 = 50k unseen) | New 50k: recall 97.82%, positive every-match 93.625%, median/p95/p99/max 542/678/731/891 | `local_m1/benchmark_metrics.json` |
| Local full-fold ETA | 4.07 h, or 5.08 h with 25% headroom. Peak stage RSS 2.05 GiB. 2.3 GiB new shards with headroom. | `local_m1/benchmark_eta.json` |

## Recommended instance

- 16 vCPU and 64 GiB RAM, for example GCP `n2-highmem-16` or AWS `r6i.4xlarge`. At minimum use 8 vCPU with 32 GiB.
- Six workers at about 2.5 GiB each is about 15 GiB total.
- 30 GB disk: 1.2 GB training data plus about 3 GB of shards.
- Colab high-RAM CPU also works: `!bash run_phase1c_parallel.sh`. Keep the runtime alive, or rerun the script after a disconnect to resume.
- No GPU. This stage is sparse CPU/memory-bandwidth bound: 1→8 threads improved the address query by only about 20%.

The projected critical path at M1 per-core speed is the S2 address-char worker at about 87 min. Adding the on-machine pilot (about 30 min) and evaluation gives about 2–2.5 h in total.

## Steps

1. **Copy the tree.** Test data and local shards stay behind:
   ```bash
   rsync -a --exclude 'student_resource/dataset/test' --exclude 'artifacts/phase1c/work' \
     --exclude '.work' --exclude '__MACOSX' --exclude 'outputs' \
     "business-entity-resolution/" VM:business-entity-resolution/
   ```
   Required on the VM:
   - `student_resource/dataset/train/`
   - `artifacts/folds.tsv`
   - `artifacts/phase1a/pilot_candidates.npz` and `pilot_metrics.json`
   - `artifacts/phase1b/`
   - `configs/`
   - `artifacts/phase1c/benchmark_eta.json`
   - `code/`
2. **Run** on the VM, from `code/business_entity_resolution`, with Python 3.11+:
   ```bash
   nohup bash run_phase1c_parallel.sh > ../../artifacts/phase1c/cloud_run.log 2>&1 &
   ```
   The script does the following:
   1. Installs the pinned requirements and runs the tests.
   2. Reproduces the 25k pilot on this hardware, and stops if it is not exact.
   3. Launches six disjoint workers: {S2, S3} × {address_char, name_char, other routes}. Country partitions are discovered dynamically.
   4. Runs a final unfiltered pass that verifies every shard SHA-256, evaluates all 441,467 S1, and writes the outputs.

   If anything is interrupted, rerun the same command. Completed tasks are skipped.
3. **Copy results back:**
   ```bash
   rsync -a VM:business-entity-resolution/artifacts/phase1c/ "business-entity-resolution/artifacts/phase1c/"
   ```
   This brings back:
   - `fold0_metrics.json`, `fold0_report.md`, `run_manifest.json`
   - `pilot_reproduction.json` (the cloud-hardware version; the M1 copy stays in `local_m1/`)
   - `work_cloud/`, which holds the route shards retained for Phase 2 (about 2.7 GiB).

   Local disk had 15.8 GiB free.

## Acceptance

`fold0_metrics.json → gate.passed` must be true, and every check must be listed individually:

- recall ≥ 97.292%;
- positive every-match ≥ 92.5%;
- p95 ≤ 800;
- no country, source, non-ASCII, or missing-address slice more than 2 pp below the pilot;
- pilot reproduced exactly.

Only then does Phase 2 begin.
