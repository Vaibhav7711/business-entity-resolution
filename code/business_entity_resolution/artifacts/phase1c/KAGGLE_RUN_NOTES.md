# Phase 1C full-fold run: provenance and interpretation notes

`fold0_metrics.json`, `fold0_report.md` and `run_manifest.json` in this folder are copies of the Kaggle full-fold outputs. The complete Kaggle download is preserved unmodified in `artifacts/phase1c_kaggle/`, and its SHA-256 hashes are listed in `run_manifest.json → outputs_sha256`.

## Platform

- Kaggle CPU: x86_64, 4 vCPU, 31.3 GiB RAM, Python 3.12.13.
- Training inputs were SHA-256-verified identical to the local copies.
- This is the **reference platform**. Phase 2 training pairs and final test candidates must also come from it.

## Pilot gate

`basis = platform_float_tolerance` (see `artifacts/phase1c_kaggle/pilot_reproduction.json`).

The `pilot_reproduction.json` in *this* folder is the Apple M1 run, which is exact. Its copy is in `local_m1/`. The float-tolerance rationale is in `colab_pilot_diagnosis.md`.

Shard 0 of the full run versus Phase 1B:
- The 12 non-float route shards are identical.
- The pilot differs by 1 true link out of 86,372.
- Median, p95, p99 and max candidate counts are identical.

## Cross-check with Phase 2A

The Phase 2A benchmark candidates were generated separately on Kaggle (`artifacts/phase2a_kaggle/phase1c_bench/`). Their shard-0 pilot comparison is identical to the full run's. So Phase 2A used the same candidates as the full run.

## Resource figures

The run used 6 parallel shard workers.

- **Elapsed wall clock:** about 3.8 h, from 16:19 to 20:09 Kaggle time, per `run_manifest.json → invocations`. The pilot took about 0.34 h. The workers ran for 0.9–3.5 h each, with the S3 address worker as the critical path. The final evaluation took 0.05 h.
- **"All Phase 1C invocations: 13.58 h"** in `fold0_report.md` adds the parallel workers together. It is worker-hours, not elapsed time.
- **"2566.9 S1/s"** in `fold0_report.md` covers only the final evaluation invocation. Over the elapsed time, end-to-end throughput was about 32 S1/s.

The report generator has since been fixed to print the elapsed wall clock.

## Other figures

- **Peak process RSS:** 4.93 GiB, in a single invocation (the pilot); the address workers peaked at 4.7–4.9 GiB.
- **Route shards:** about 2.26 GiB. They remain in the Kaggle notebook output (`work/shards`) and were not downloaded; the content hash of every task is in `run_manifest.json → shard_content_sha256`.
- **Max candidates per S1:** 1,242, in the US, from large generic exact-name buckets. The pilot maximum was 791. p95 is 679 and p99 is 731.
