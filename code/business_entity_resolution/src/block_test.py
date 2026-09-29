"""Frozen B+C+E blocker inference on the TEST split: no labels, checkpointed, shardable.

Run from code/business_entity_resolution, one shard range per machine:

    python3 -m src.block_test --config ../../configs/phase1c_fold0.json --shards 0-23 \
        --only-source 2 --only-route address_char          # parallel worker (no finalize)
    python3 -m src.block_test --config ../../configs/phase1c_fold0.json --shards 0-23 --finalize

Test S1 records are ordered by BLAKE2b(seed:ID) and cut into ``shard_size`` shards,
exactly like the validation fold. The route code is the gate-validated Phase 1C code
(``generate_tasks``); the target corpus searched is test S2/S3, partitioned by the
dynamic normalized country (France included). Ground truth, folds, and training
files are never opened. ``--finalize`` verifies every task of the range and writes
an unlabeled diagnostics manifest (candidate counts by country, partition sizes).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluate_blocking import ROOT
from .evaluate_phase1c import (
    DENSE_ROUTES, ROUTES, SCORED_RAGGED_ROUTES, StopRequested, Workspace, append_jsonl, atomic_write_json,
    candidate_union, code_state, count_summary, country_slug, environment, generate_tasks, log,
    peak_rss_bytes, rank_route, read_jsonl, sha256_file, sha256_json, validate_config,
)
from .normalization import has_non_ascii, normalize_text

TEST_FILES = ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv")


def ordered_test_ids(test_dir: Path, seed: int) -> list[str]:
    """All test S1 IDs by ascending BLAKE2b(seed:ID) (same rule as the validation fold)."""
    scored = []
    with (test_dir / "test_source1.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            entity_id = row["entity_id"]
            scored.append((hashlib.blake2b(f"{seed}:{entity_id}".encode(), digest_size=8).digest(), entity_id))
    ids = [entity_id for _, entity_id in sorted(scored)]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate test S1 IDs")
    return ids


def load_test_queries(test_dir: Path, ids: list[str], positions: list[int]) -> dict:
    """Normalized S1 fields for the requested positions only (others stay None)."""
    wanted = {ids[p]: p for p in positions}
    n = len(ids)
    fields = {name: [None] * n for name in ("country", "country_key", "name", "address")}
    non_ascii = np.zeros(n, dtype=np.bool_)
    with (test_dir / "test_source1.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            i = wanted.get(row["entity_id"])
            if i is None:
                continue
            fields["country"][i] = row["country"]
            fields["country_key"][i] = normalize_text(row["country"])
            fields["name"][i] = normalize_text(row["business_name"])
            fields["address"][i] = normalize_text(row["business_address"])
            non_ascii[i] = has_non_ascii(row["business_name"]) or has_non_ascii(row["business_address"])
    if any(fields["country"][p] is None for p in positions):
        raise RuntimeError("Requested test S1 positions missing from test_source1.tsv")
    return {**fields, "non_ascii": non_ascii}


def parse_shards(text: str, n_shards: int) -> list[int]:
    first, _, last = text.partition("-")
    start, end = int(first), int(last or first)
    if not 0 <= start <= end < n_shards:
        raise ValueError(f"--shards {text} outside 0-{n_shards - 1}")
    return list(range(start, end + 1))


def check_test_state(ws: Workspace, config: dict, ordered: list[str]) -> dict:
    state = {"run_id": f"{config['run_id']}_test", "split": "test",
             "algorithm_sha256": sha256_json(config["algorithm"]),
             "order_sha256": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
             "n_s1": len(ordered), "shard_size": config["algorithm"]["shard_size"]}
    if ws.state.exists():
        existing = json.loads(ws.state.read_text())
        if {key: existing.get(key) for key in state} != state:
            raise RuntimeError(f"Work directory {ws.work} belongs to a different config or test order")
    else:
        atomic_write_json(ws.state, {**state, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return state


def shard_unions(ws: Workspace, shard: int, positions_by_key: dict[str, np.ndarray], rrf_constant: int):
    """Yield (position, union IDs) for every S1 of one shard from its route files (hash-verified)."""
    for key, positions in sorted(positions_by_key.items()):
        slug = country_slug(key)
        loaded = {}
        for route in ROUTES:
            for source in (2, 3):
                path = ws.task_path(source, slug, route, shard)
                sidecar = json.loads(path.with_suffix(".json").read_text())
                if sha256_file(path) != sidecar["file_sha256"]:
                    raise RuntimeError(f"Checksum mismatch for {path}")
                with np.load(path, allow_pickle=False) as data:
                    arrays = {name: data[name] for name in data.files}
                if not np.array_equal(arrays["positions"], positions):
                    raise RuntimeError(f"Query order mismatch in {path}")
                loaded[route, source] = arrays
        for local, position in enumerate(positions):
            ranked = []
            for route in ROUTES:
                parts = []
                for source in (2, 3):
                    arrays = loaded[route, source]
                    if route in DENSE_ROUTES:
                        count = int(arrays["counts"][local])
                        parts.append((arrays["ids"][local, :count], arrays["scores"][local, :count]))
                    else:
                        left, right = arrays["indptr"][local:local + 2]
                        parts.append((arrays["ids"][left:right],
                                      arrays["scores"][left:right] if route in SCORED_RAGGED_ROUTES else None))
                ranked.append(rank_route(parts))
            uids, _, _, _ = candidate_union(ranked, rrf_constant)
            yield int(position), uids


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run_test(config: dict, root: Path, test_dir: Path, shards_text: str, work_dir: Path, output_dir: Path, *,
             stop_after_tasks: int | None = None, only_sources: list[int] | None = None,
             only_countries: list[str] | None = None, only_routes: list[str] | None = None,
             finalize: bool = False) -> dict:
    started = time.perf_counter()
    validate_config(config)
    algo, ops = config["algorithm"], config["operational"]
    ws = Workspace(work_dir)
    ordered = ordered_test_ids(test_dir, algo["seed"])
    state = check_test_state(ws, config, ordered)
    size = algo["shard_size"]
    n_shards = (len(ordered) + size - 1) // size
    active = parse_shards(shards_text, n_shards)
    positions = [p for shard in active for p in range(shard * size, min((shard + 1) * size, len(ordered)))]
    queries = load_test_queries(test_dir, ordered, positions)
    shard_positions: dict[int, dict[str, np.ndarray]] = {}
    for shard in active:
        grouped = defaultdict(list)
        for position in range(shard * size, min((shard + 1) * size, len(ordered))):
            grouped[queries["country_key"][position]].append(position)
        shard_positions[shard] = {key: np.asarray(values, dtype=np.int64) for key, values in grouped.items()}
    keys = sorted({key for groups in shard_positions.values() for key in groups})
    filtered = bool(only_sources or only_countries or only_routes)
    invocation = {"invocation_id": uuid.uuid4().hex[:12], "scope": f"test:{shards_text}", "split": "test",
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
                  "stop_after_tasks": stop_after_tasks, "tasks_completed": 0, "tasks_skipped": 0,
                  "partitions_scanned": [], "partitions_skipped": [], "fits": [], "cpu_model": cpu_model(),
                  "filters": {"sources": only_sources, "countries": only_countries, "routes": only_routes}}
    log(f"test {shards_text}: {len(ordered):,} test S1 in {n_shards} shards; {len(positions):,} S1 in range; "
        f"dynamic country keys: {keys}")
    if not finalize:
        try:
            generate_tasks(ws, algo, ops, lambda source: test_dir / f"test_source{source}.tsv", active,
                           shard_positions, queries, set(), invocation, stop_after_tasks=stop_after_tasks,
                           only_sources=only_sources, only_countries=only_countries, only_routes=only_routes)
        except StopRequested:
            invocation.update(status="stopped", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                              wall_seconds=time.perf_counter() - started, peak_rss=peak_rss_bytes())
            append_jsonl(ws.invocations, invocation)
            return {"status": "stopped", "invocation": invocation}
        status = "generated" if filtered else "generated_all"
        invocation.update(status=status, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          wall_seconds=time.perf_counter() - started, peak_rss=peak_rss_bytes())
        append_jsonl(ws.invocations, invocation)
        log(f"test {shards_text}: {invocation['tasks_completed']} new tasks, {invocation['tasks_skipped']} skipped")
        if filtered:
            return {"status": status, "invocation": invocation}

    missing = [f"S{source}/{key}/{route}/shard{shard}" for shard in active for key in shard_positions[shard]
               for route in ROUTES for source in algo["target_sources"]
               if not ws.task_done(source, country_slug(key), route, shard)]
    if missing:
        raise RuntimeError(f"{len(missing)} tasks missing for range {shards_text}, e.g. {missing[:5]}; rerun workers")
    counts, country_of = {}, {}
    for shard in active:
        for position, uids in shard_unions(ws, shard, shard_positions[shard], ops["rrf_constant"]):
            counts[position] = len(uids)
            country_of[position] = queries["country"][position]
    by_country = defaultdict(list)
    for position, count in counts.items():
        by_country[country_of[position]].append(count)
    partitions = {}
    for key in keys:
        for source in algo["target_sources"]:
            meta = json.loads(ws.partition_meta_path(source, country_slug(key)).read_text())
            partitions[f"S{source}/{key}"] = meta["targets"]
    all_counts = np.asarray(list(counts.values()), dtype=np.int64)
    manifest = {
        "status": "complete", "split": "test", "shards": [active[0], active[-1]], "state": state,
        "s1_in_range": len(positions), "s1_by_country": {k: len(v) for k, v in sorted(by_country.items())},
        "target_partition_sizes": partitions,
        "candidate_count": count_summary(all_counts),
        "zero_candidate_rate": float(np.mean(all_counts == 0)),
        "candidate_count_by_country": {k: count_summary(np.asarray(v)) | {"zero_candidate_rate": float(np.mean(np.asarray(v) == 0))}
                                       for k, v in sorted(by_country.items())},
        "tasks": {str(path.relative_to(ws.work)): json.loads(path.read_text())["content_sha256"]
                  for shard in active for path in sorted(ws.shards.rglob(f"shard{shard:03d}.json"))},
        "invocations": [row for row in read_jsonl(ws.invocations) if row.get("scope") == f"test:{shards_text}"],
        "test_input_sha256": {name: sha256_file(test_dir / name) for name in TEST_FILES},
        "environment": environment() | {"cpu_model": cpu_model()}, "code_state": code_state(ROOT),
        "labels_read": False,
        "note": "Unlabeled diagnostics only (candidate counts, partition sizes); no test record was used for design.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"test_range_{active[0]:03d}_{active[-1]:03d}_manifest.json"
    atomic_write_json(path, manifest)
    log(f"test {shards_text}: finalized {len(counts):,} S1; candidates median "
        f"{manifest['candidate_count']['median']:g}, p95 {manifest['candidate_count']['p95']:g}, zero-candidate "
        f"{manifest['zero_candidate_rate']:.4%}; by country {manifest['s1_by_country']} -> {path}")
    return {"status": "complete", "manifest": manifest, "counts": counts}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="frozen blocker config (phase1c_fold0.json)")
    parser.add_argument("--shards", required=True, help="inclusive shard range, e.g. 0-23")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "student_resource/dataset/test")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts/test_blocking/work")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/test_blocking")
    parser.add_argument("--only-source", type=int, action="append", choices=(2, 3))
    parser.add_argument("--only-country", action="append")
    parser.add_argument("--only-route", action="append", choices=ROUTES)
    parser.add_argument("--stop-after-tasks", type=int)
    parser.add_argument("--finalize", action="store_true", help="verify the range and write its manifest")
    args = parser.parse_args(argv)
    config = json.loads(args.config.resolve().read_text())
    run_test(config, ROOT, args.test_dir, args.shards, args.work_dir, args.output_dir,
             stop_after_tasks=args.stop_after_tasks, only_sources=args.only_source,
             only_countries=[normalize_text(v) for v in args.only_country] if args.only_country else None,
             only_routes=args.only_route, finalize=args.finalize)


if __name__ == "__main__":
    main()
