"""Phase 1C: checkpointed, resource-bounded full-fold evaluation of the frozen B+C+E blocker.

Run from code/business_entity_resolution:

    python3 -m src.evaluate_phase1c --config ../../configs/phase1c_fold0.json --scope pilot
    python3 -m src.evaluate_phase1c --config ../../configs/phase1c_fold0.json --scope benchmark
    python3 -m src.evaluate_phase1c --config ../../configs/phase1c_fold0.json --scope full

Validation-fold S1 records are ordered by the Phase 1A/1B BLAKE2b rank, so shard 0 is
exactly the 25,000-record pilot. Work is checkpointed per target source x dynamic
country x route x S1 shard; each target representation is fitted once per invocation
and completed shards are skipped on resume. Only training files are read and gold
links are never added to candidate sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .blocking import encode_id, fit_hashed_tfidf, query_hashed_tfidf
from .evaluate_blocking import PeakRSS, ROOT
from .normalization import has_non_ascii, normalize_text
from .phase1b_routes import (
    LEGAL_SUFFIXES, RareTokenIndex, build_key_index, fit_word_tfidf, name_tokens,
    strip_legal_suffix,
)

ROUTES = ("exact_name", "name_char", "address_char", "name_word", "rare_name", "suffix_exact")
ROUTE_BITS = {route: 1 << i for i, route in enumerate(ROUTES)}
DENSE_ROUTES = frozenset({"name_char", "address_char", "name_word"})
SCORED_RAGGED_ROUTES = frozenset({"rare_name"})
# Cumulative route sets: Phase 1A union, then the Phase 1B selections B, C, E in order.
STEPS = (
    ("phase1a_union", 0b000111),
    ("plus_B_name_word", 0b001111),
    ("plus_C_rare_name", 0b011111),
    ("plus_E_suffix_exact", 0b111111),
)
SELECTED = "plus_E_suffix_exact"
BUCKETS = ("0", "1", "2", "3", "4", "5", "6+")
PACKAGES = ("numpy", "scipy", "scikit-learn", "psutil", "sparse-dot-topn")
SCOPES = ("pilot", "benchmark", "full")


class StopRequested(Exception):
    """Raised after the requested number of newly completed tasks."""


def ensure_memory(ops: dict, label: str, *, poll_seconds: float = 15.0) -> None:
    """Wait for free RAM before a fit or shard task; stop cleanly if it never recovers.

    Protects small machines from system-wide out-of-memory crashes. It only
    delays work, so candidate outputs are unaffected.
    """
    import psutil

    needed = ops.get("min_available_memory_gib", 0) * 2**30
    max_wait = ops.get("max_memory_wait_seconds", 1800)
    waited = 0.0
    while needed and psutil.virtual_memory().available < needed:
        if waited == 0:
            log(f"{label}: waiting for free memory ({psutil.virtual_memory().available/2**30:.2f} GiB available, "
                f"{needed/2**30:.2f} GiB required)")
        if waited >= max_wait:
            log(f"{label}: memory did not recover in {max_wait:.0f}s; stopping cleanly (rerun to resume)")
            raise StopRequested
        time.sleep(poll_seconds)
        waited += poll_seconds


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Hashing and atomic persistence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    """Hash array names, dtypes, shapes, and bytes; independent of zip timestamps."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}|{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")
    os.replace(temporary, path)


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        np.savez(file, **arrays)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def country_slug(country_key: str) -> str:
    """Filesystem-safe, collision-resistant name for a dynamic country key."""
    readable = re.sub(r"[^0-9a-z]+", "_", country_key.encode("ascii", "ignore").decode().lower()).strip("_")
    digest = hashlib.blake2b(country_key.encode(), digest_size=4).hexdigest()
    return f"{readable[:32] or 'country'}-{digest}"


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


# ---------------------------------------------------------------------------
# Inputs


def ordered_fold_ids(folds_path: Path, fold: int, seed: int) -> list[str]:
    """All fold S1 IDs by ascending BLAKE2b(seed:ID); the Phase 1A sample is a prefix."""
    scored = []
    with folds_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if int(row["fold"]) == fold:
                entity_id = row["source1_entity_id"]
                digest = hashlib.blake2b(f"{seed}:{entity_id}".encode(), digest_size=8).digest()
                scored.append((digest, entity_id))
    return [entity_id for _, entity_id in sorted(scored)]


def load_queries(train_dir: Path, ids: list[str]) -> dict:
    """Normalized S1 fields aligned to ``ids``; only the training S1 file is read."""
    position = {entity_id: i for i, entity_id in enumerate(ids)}
    n = len(ids)
    country, country_key, names, addresses = [None] * n, [None] * n, [None] * n, [None] * n
    non_ascii = np.zeros(n, dtype=np.bool_)
    with (train_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            i = position.get(row["entity_id"])
            if i is None:
                continue
            country[i] = row["country"]
            country_key[i] = normalize_text(row["country"])
            names[i] = normalize_text(row["business_name"])
            addresses[i] = normalize_text(row["business_address"])
            non_ascii[i] = has_non_ascii(row["business_name"]) or has_non_ascii(row["business_address"])
    if any(value is None for value in country):
        raise RuntimeError("Fold S1 IDs missing from the training S1 file")
    return {"country": country, "country_key": country_key, "name": names,
            "address": addresses, "non_ascii": non_ascii}


def load_truth(train_dir: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth links aligned to ``ids`` as CSR (indptr, sorted encoded target IDs)."""
    position = {entity_id: i for i, entity_id in enumerate(ids)}
    rows: list[list[int] | None] = [None] * len(ids)
    with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            i = position.get(row["source1_entity_id"])
            if i is not None:
                rows[i] = sorted({encode_id(value) for value in row["matched_entity_ids"].split(",") if value})
    if any(row is None for row in rows):
        raise RuntimeError("Fold S1 IDs missing from the training ground truth")
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    values = np.fromiter((value for row in rows for value in row), dtype=np.int64, count=int(indptr[-1]))
    return indptr, values


def scan_target_partition(path: Path, country_key: str, truth_targets: set[int]) -> dict:
    """Stream one training target file; keep one dynamic-country partition.

    Metadata for every fold true target is collected from all rows, including
    rows outside this partition, so cross-country gold links remain visible.
    """
    ids, names, addresses = [], [], []
    meta: dict[int, tuple[bool, bool, str]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            key = normalize_text(row["country"])
            target_id = encode_id(row["entity_id"])
            if target_id in truth_targets:
                meta[target_id] = (
                    has_non_ascii(row["business_name"]) or has_non_ascii(row["business_address"]),
                    not row["business_address"].strip(), key,
                )
            if key != country_key:
                continue
            ids.append(target_id)
            names.append(normalize_text(row["business_name"]))
            addresses.append(normalize_text(row["business_address"]))
    if ids and (max(ids) > np.iinfo(np.uint32).max or min(ids) == 0):
        raise RuntimeError("Encoded target IDs do not fit the uint32 shard format")
    return {"ids": ids, "ids_np": np.asarray(ids, dtype=np.uint32), "name": names,
            "address": addresses, "meta": meta}


# ---------------------------------------------------------------------------
# Route fitting and bounded querying


def fit_route(route: str, targets: dict, queries: dict, positions: np.ndarray, algo: dict):
    """Fit one retrieval representation on one complete target partition."""
    if not targets["ids"]:
        return None
    if route == "exact_name":
        wanted = {queries["name"][i] for i in positions}
        return build_key_index(targets["name"], targets["ids"], wanted)
    if route == "suffix_exact":
        wanted = {strip_legal_suffix(queries["name"][i]) for i in positions}
        return build_key_index((strip_legal_suffix(name) for name in targets["name"]), targets["ids"], wanted)
    if route == "rare_name":
        wanted = set().union(*(name_tokens(queries["name"][i]) for i in positions))
        return RareTokenIndex(targets["name"], targets["ids"], tokenize=name_tokens, wanted=wanted,
                              max_document_frequency=algo["rare_name_max_document_frequency"])
    if route == "name_word":
        return fit_word_tfidf(targets["name"], n_features=algo["word_hash_features"],
                              max_document_frequency=algo["word_max_document_frequency"])
    field = "name" if route == "name_char" else "address"
    return fit_hashed_tfidf(targets[field], analyzer="char", ngram_range=tuple(algo["char_ngram_range"]),
                            n_features=algo["char_hash_features"],
                            max_document_frequency=algo["char_max_document_frequency"])


def fill_topk(product, target_ids: np.ndarray, out_ids: np.ndarray, out_scores: np.ndarray,
              out_counts: np.ndarray) -> None:
    """Write sparse top-k rows ordered by (-score, target ID), as in Phase 1A/1B."""
    lengths = np.diff(product.indptr)
    rows = np.repeat(np.arange(len(lengths)), lengths)
    ids = target_ids[product.indices]
    scores = product.data.astype(np.float32, copy=False)
    order = np.lexsort((ids, -scores, rows))
    positions = np.arange(len(ids)) - np.repeat(product.indptr[:-1], lengths)
    out_ids[rows, positions] = ids[order]
    out_scores[rows, positions] = scores[order]
    out_counts[:len(lengths)] = lengths


def query_route(route: str, index, targets: dict, queries: dict, positions: np.ndarray,
                algo: dict, ops: dict) -> dict[str, np.ndarray]:
    """Candidate arrays for one route, source/country partition, and S1 shard."""
    m = len(positions)
    if route in DENSE_ROUTES:
        k = algo["word_top_k_per_source"] if route == "name_word" else algo["char_top_k_per_source"]
        ids = np.zeros((m, k), dtype=np.uint32)
        scores = np.zeros((m, k), dtype=np.float32)
        counts = np.zeros(m, dtype=np.uint16)
        if index is not None:
            field = "address" if route == "address_char" else "name"
            texts = [queries[field][i] for i in positions]
            batch = ops["query_batch_size"]
            for start in range(0, m, batch):
                product = query_hashed_tfidf(index, texts[start:start + batch], k=k, threads=ops["threads"])
                fill_topk(product, targets["ids_np"], ids[start:], scores[start:], counts[start:])
        return {"positions": positions.astype(np.int64), "ids": ids, "scores": scores, "counts": counts}
    rows: list[list[int]] = []
    row_scores: list[list[float]] = []
    for i in positions:
        name = queries["name"][i]
        if index is None:
            rows.append([])
        elif route == "exact_name":
            rows.append(list(index.get(name, [])) if name else [])
        elif route == "suffix_exact":
            key = strip_legal_suffix(name)
            rows.append(list(index.get(key, [])) if key else [])
        else:
            hits = index.query(name, max_query_tokens=algo["rare_name_max_query_tokens"],
                               k=algo["rare_name_top_k_per_source"])
            rows.append([value for value, _ in hits])
            row_scores.append([score for _, score in hits])
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=m)
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    arrays = {
        "positions": positions.astype(np.int64), "indptr": indptr,
        "ids": np.fromiter((v for row in rows for v in row), dtype=np.uint32, count=int(indptr[-1])),
    }
    if route in SCORED_RAGGED_ROUTES:
        arrays["scores"] = np.fromiter((s for row in row_scores for s in row), dtype=np.float32,
                                       count=int(indptr[-1])) if row_scores else np.zeros(0, np.float32)
    return arrays


# ---------------------------------------------------------------------------
# Per-S1 union and link accounting


def rank_route(parts: list[tuple[np.ndarray, np.ndarray | None]]) -> np.ndarray:
    """One route's ranked candidate IDs across both target sources.

    Scored routes are ordered by (-score, ID); exact lookups by ID. Every
    distinct target ID is kept, even when several targets share their text.
    """
    ids = np.concatenate([part[0] for part in parts]).astype(np.int64, copy=False)
    if parts[0][1] is None:
        return np.unique(ids)
    scores = np.concatenate([part[1] for part in parts])
    return ids[np.lexsort((ids, -scores))]


def candidate_union(ranked: list[np.ndarray], rrf_constant: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unique union with route-membership bits and a deterministic RRF order.

    Returns sorted unique IDs, their route bits, their 0-based RRF rank, and
    the RRF score. RRF ties are broken by target ID.
    """
    lengths = [len(row) for row in ranked]
    all_ids = np.concatenate(ranked) if ranked else np.zeros(0, np.int64)
    if not len(all_ids):
        empty = np.zeros(0, np.int64)
        return empty, np.zeros(0, np.uint8), empty, np.zeros(0, np.float64)
    route_bits = np.repeat(np.asarray([1 << i for i in range(len(ranked))], dtype=np.uint8), lengths)
    ranks = np.concatenate([np.arange(1, length + 1) for length in lengths])
    uids, inverse = np.unique(all_ids, return_inverse=True)
    bits = np.zeros(len(uids), dtype=np.uint8)
    np.bitwise_or.at(bits, inverse, route_bits)
    rrf = np.bincount(inverse, weights=1.0 / (rrf_constant + ranks), minlength=len(uids))
    order = np.lexsort((uids, -rrf))
    rank = np.empty(len(uids), dtype=np.int64)
    rank[order] = np.arange(len(uids))
    return uids, bits, rank, rrf


# ---------------------------------------------------------------------------
# Metrics


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def count_summary(counts: np.ndarray) -> dict:
    if not len(counts):
        return {}
    return {
        "mean": float(np.mean(counts)), "median": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)), "p95": float(np.percentile(counts, 95)),
        "p99": float(np.percentile(counts, 99)), "max": int(counts.max()), "total": int(counts.sum()),
    }


def bucket_of(length: np.ndarray) -> np.ndarray:
    return np.asarray([BUCKETS[min(int(value), 6)] for value in length])


def summarize(ev: dict, mask: int, counts: np.ndarray, *, detailed: bool = False,
              recall_at_k: tuple[int, ...] = ()) -> dict:
    """Blocker metrics for one route mask from per-S1 and per-link arrays."""
    n = len(ev["country"])
    hit = (ev["link_bits"] & mask) != 0
    link_s1 = ev["link_s1"]
    truth_len = ev["truth_len"]
    positive = truth_len > 0
    misses = np.bincount(link_s1[~hit], minlength=n)
    every = positive & (misses == 0)
    total_links = int(len(hit))
    countries = np.asarray(ev["country"], dtype=object)
    link_country = countries[link_s1]
    result = {
        "evaluated_s1": n,
        "true_links": total_links,
        "retrieved_true_links": int(hit.sum()),
        "positive_link_recall": ratio(int(hit.sum()), total_links),
        # Same expression order as Phase 1B so pilot values compare exactly.
        "positive_s1_with_every_true_match_pct": 100 * int(every.sum()) / int(positive.sum()) if positive.any() else None,
        "all_s1_every_match_pct": 100 * float(np.mean(every | ~positive)) if n else None,
        "zero_candidate_rate": float(np.mean(counts == 0)) if n else None,
        "candidate_count": count_summary(counts),
        "candidate_reduction_ratio": 1 - int(counts.sum()) / int(ev["pool"].sum()) if n else None,
        "recall_by_country": {
            country: ratio(int(hit[link_country == country].sum()), int((link_country == country).sum()))
            for country in sorted(set(ev["country"]))
        },
        "recall_by_source": {
            source: ratio(int(hit[(ev["link_ids"] & 1) == parity].sum()), int(((ev["link_ids"] & 1) == parity).sum()))
            for source, parity in (("S2", 0), ("S3", 1))
        },
        "non_ascii_recall": ratio(int(hit[ev["link_non_ascii"]].sum()), int(ev["link_non_ascii"].sum())),
        "missing_target_address_recall": ratio(int(hit[ev["link_missing"]].sum()), int(ev["link_missing"].sum())),
    }
    if not detailed:
        return result
    buckets = bucket_of(truth_len)
    result["by_match_count_bucket"] = {}
    for bucket in BUCKETS:
        selected = buckets == bucket
        link_selected = selected[link_s1]
        result["by_match_count_bucket"][bucket] = {
            "s1": int(selected.sum()),
            "true_links": int(link_selected.sum()),
            "link_recall": ratio(int(hit[link_selected].sum()), int(link_selected.sum())),
            "every_match_pct": 100 * float(np.mean(every[selected] | ~positive[selected])) if selected.any() else None,
            "zero_candidate_rate": float(np.mean(counts[selected] == 0)) if selected.any() else None,
            "candidate_mean": float(np.mean(counts[selected])) if selected.any() else None,
        }
    result["by_country"] = {}
    for country in sorted(set(ev["country"])):
        selected = countries == country
        link_selected = selected[link_s1]
        result["by_country"][country] = {
            "s1": int(selected.sum()), "true_links": int(link_selected.sum()),
            "link_recall": ratio(int(hit[link_selected].sum()), int(link_selected.sum())),
            "positive_every_match_pct": 100 * int(every[selected].sum()) / int(positive[selected].sum())
            if positive[selected].any() else None,
            "candidate_count": count_summary(counts[selected]),
            "zero_candidate_rate": float(np.mean(counts[selected] == 0)),
            "non_ascii_recall": ratio(int(hit[link_selected & ev["link_non_ascii"]].sum()),
                                      int((link_selected & ev["link_non_ascii"]).sum())),
            "missing_target_address_recall": ratio(int(hit[link_selected & ev["link_missing"]].sum()),
                                                   int((link_selected & ev["link_missing"]).sum())),
        }
    result["positive_s1"] = int(positive.sum())
    result["singleton_s1"] = int((~positive).sum())
    result["cross_country_true_links"] = int(ev["link_cross_country"].sum())
    result["cross_country_true_links_retrieved"] = int(hit[ev["link_cross_country"]].sum())
    result["non_ascii_true_links"] = int(ev["link_non_ascii"].sum())
    result["missing_target_address_true_links"] = int(ev["link_missing"].sum())
    result["candidate_count_by_source"] = {"S2": count_summary(ev["counts"]["S2"]),
                                           "S3": count_summary(ev["counts"]["S3"])}
    if recall_at_k:
        rank = ev["link_rrf_rank"]
        result["recall_at_k_rrf"] = {str(k): ratio(int((hit & (rank < k)).sum()), total_links) for k in recall_at_k}
    return result


def route_contributions(ev: dict) -> dict:
    bits = ev["link_bits"]
    total = len(bits)
    result = {"route_alone": {}, "unique_true_links": {}, "cumulative": {}}
    for route, bit in ROUTE_BITS.items():
        hits = int(((bits & bit) != 0).sum())
        counts = ev["counts"][route]
        result["route_alone"][route] = {
            "retrieved_true_links": hits, "link_recall": ratio(hits, total),
            "candidate_total": int(counts.sum()), "candidate_mean": float(np.mean(counts)) if len(counts) else None,
            "candidate_median": float(np.median(counts)) if len(counts) else None,
        }
        result["unique_true_links"][route] = int((bits == bit).sum())
    previous_hits = previous_total = 0
    for step, mask in STEPS:
        hits = int(((bits & mask) != 0).sum())
        candidates = int(ev["counts"][step].sum())
        result["cumulative"][step] = {
            "retrieved_true_links": hits, "link_recall": ratio(hits, total), "candidate_total": candidates,
            "marginal_true_links": hits - previous_hits, "marginal_candidates": candidates - previous_total,
        }
        previous_hits, previous_total = hits, candidates
    return result


# ---------------------------------------------------------------------------
# Run orchestration


class Workspace:
    def __init__(self, work: Path):
        self.work = work
        self.shards = work / "shards"
        self.meta = work / "meta"
        self.timing = work / "timing.jsonl"
        self.invocations = work / "invocations.jsonl"
        self.state = work / "state.json"

    def task_path(self, source: int, slug: str, route: str, shard: int) -> Path:
        return self.shards / f"S{source}" / slug / route / f"shard{shard:03d}.npz"

    def task_done(self, source: int, slug: str, route: str, shard: int) -> bool:
        path = self.task_path(source, slug, route, shard)
        return path.exists() and path.with_suffix(".json").exists()

    def partition_meta_path(self, source: int, slug: str) -> Path:
        return self.meta / f"S{source}_{slug}.json"

    def truth_meta_path(self, source: int) -> Path:
        return self.meta / f"S{source}_truth_targets.npz"


def validate_config(config: dict) -> None:
    algo = config["algorithm"]
    if tuple(algo["routes"]) != ROUTES:
        raise ValueError(f"Configured routes differ from the frozen B+C+E blocker: {algo['routes']}")
    if config["phase1b_selected_mask"] != 22:
        raise ValueError("Phase 1C evaluates only the selected Phase 1B mask 22 (B+C+E)")
    if set(algo["legal_suffix_spellings"]) != LEGAL_SUFFIXES:
        raise ValueError("Configured suffixes differ from the supplied problem statement")
    if list(algo["target_sources"]) != [2, 3]:
        raise ValueError("Target sources must be S2 and S3")
    if Path(config["paths"]["train_dir"]).name != "train":
        raise ValueError("Phase 1C reads only the training directory")


def check_state(ws: Workspace, config: dict, ordered: list[str]) -> dict:
    state = {
        "run_id": config["run_id"],
        "algorithm_sha256": sha256_json(config["algorithm"]),
        "order_sha256": hashlib.sha256("\n".join(ordered).encode()).hexdigest(),
        "fold_size": len(ordered),
        "shard_size": config["algorithm"]["shard_size"],
    }
    if ws.state.exists():
        existing = json.loads(ws.state.read_text())
        existing_core = {key: existing.get(key) for key in state}
        if existing_core != state:
            raise RuntimeError(f"Work directory {ws.work} belongs to a different config or fold order: "
                               f"{existing_core} != {state}")
    else:
        atomic_write_json(ws.state, {**state, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return state


def evaluate_shards(ws: Workspace, shard_ids: list[int], shard_positions: dict[int, dict[str, np.ndarray]],
                    queries: dict, truth_indptr: np.ndarray, truth_values: np.ndarray,
                    pool_by_key: dict[str, int], target_meta: dict[int, tuple[bool, bool, str]],
                    rrf_constant: int, *, verify_hashes: bool, emit_union: bool) -> dict:
    """Stream shard files, form each S1 union, and collect compact per-S1/per-link arrays."""
    sequence: list[int] = []
    per_s1 = defaultdict(list)
    per_link = defaultdict(list)
    for shard in shard_ids:
        union_rows = {"positions": [], "lengths": [], "ids": [], "bits": [], "rrf": []} if emit_union else None
        for key, positions in sorted(shard_positions[shard].items()):
            slug = country_slug(key)
            loaded = {}
            for route in ROUTES:
                for source in (2, 3):
                    path = ws.task_path(source, slug, route, shard)
                    sidecar = json.loads(path.with_suffix(".json").read_text())
                    if verify_hashes and sha256_file(path) != sidecar["file_sha256"]:
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
                uids, bits, rank, rrf = candidate_union(ranked, rrf_constant)
                sequence.append(int(position))
                for step, mask in STEPS:
                    per_s1[step].append(int(np.count_nonzero(bits & mask)))
                for route, bit in ROUTE_BITS.items():
                    per_s1[route].append(int(np.count_nonzero(bits & bit)))
                s3 = int(np.count_nonzero(uids & 1))
                per_s1["S3"].append(s3)
                per_s1["S2"].append(len(uids) - s3)
                truth = truth_values[truth_indptr[position]:truth_indptr[position + 1]]
                if len(truth):
                    where = np.searchsorted(uids, truth)
                    found = where < len(uids)
                    found[found] = uids[where[found]] == truth[found]
                    safe = np.minimum(where, max(len(uids) - 1, 0))
                    per_link["bits"].append(np.where(found, bits[safe] if len(uids) else 0, 0).astype(np.uint8))
                    per_link["rank"].append(np.where(found, rank[safe] if len(uids) else 0, np.iinfo(np.int64).max))
                if emit_union:
                    order = np.argsort(rank)
                    union_rows["positions"].append(int(position))
                    union_rows["lengths"].append(len(uids))
                    union_rows["ids"].append(uids[order].astype(np.uint32))
                    union_rows["bits"].append(bits[order])
                    union_rows["rrf"].append(rrf[order].astype(np.float32))
        if emit_union:
            lengths = np.asarray(union_rows["lengths"], dtype=np.int64)
            atomic_savez(ws.work / "union" / f"shard{shard:03d}.npz", {
                "positions": np.asarray(union_rows["positions"], dtype=np.int64),
                "indptr": np.concatenate(([0], np.cumsum(lengths))),
                "ids": np.concatenate(union_rows["ids"]) if union_rows["ids"] else np.zeros(0, np.uint32),
                "route_bits": np.concatenate(union_rows["bits"]) if union_rows["bits"] else np.zeros(0, np.uint8),
                "rrf": np.concatenate(union_rows["rrf"]) if union_rows["rrf"] else np.zeros(0, np.float32),
            })
    order = np.asarray(sequence, dtype=np.int64)
    truth_len = np.diff(truth_indptr)[order]
    link_s1 = np.repeat(np.arange(len(order)), truth_len)
    link_ids = np.concatenate([truth_values[truth_indptr[p]:truth_indptr[p + 1]] for p in order]) if len(order) else np.zeros(0, np.int64)
    missing_meta = [int(value) for value in link_ids if int(value) not in target_meta]
    if missing_meta:
        raise RuntimeError(f"Missing metadata for {len(missing_meta)} true targets")
    s1_key = [queries["country_key"][p] for p in order]
    return {
        "positions": order,
        "country": [queries["country"][p] for p in order],
        "truth_len": truth_len,
        "pool": np.asarray([pool_by_key.get(key, 0) for key in s1_key], dtype=np.int64),
        "counts": {name: np.asarray(values, dtype=np.int64) for name, values in per_s1.items()},
        "link_s1": link_s1,
        "link_ids": link_ids,
        "link_bits": np.concatenate(per_link["bits"]) if per_link["bits"] else np.zeros(0, np.uint8),
        "link_rrf_rank": np.concatenate(per_link["rank"]) if per_link["rank"] else np.zeros(0, np.int64),
        "link_non_ascii": np.asarray([bool(queries["non_ascii"][order[s]]) or target_meta[int(v)][0]
                                      for s, v in zip(link_s1, link_ids)], dtype=np.bool_),
        "link_missing": np.asarray([target_meta[int(v)][1] for v in link_ids], dtype=np.bool_),
        "link_cross_country": np.asarray([target_meta[int(v)][2] != s1_key[s]
                                          for s, v in zip(link_s1, link_ids)], dtype=np.bool_),
    }


def subset_eval(ev: dict, keep: np.ndarray) -> dict:
    """Restrict evaluation arrays to a boolean mask over evaluated S1 rows."""
    new_index = np.cumsum(keep) - 1
    link_keep = keep[ev["link_s1"]]
    return {
        "positions": ev["positions"][keep],
        "country": [value for value, flag in zip(ev["country"], keep) if flag],
        "truth_len": ev["truth_len"][keep], "pool": ev["pool"][keep],
        "counts": {name: values[keep] for name, values in ev["counts"].items()},
        "link_s1": new_index[ev["link_s1"][link_keep]],
        **{name: ev[name][link_keep] for name in ("link_ids", "link_bits", "link_rrf_rank", "link_non_ascii",
                                                   "link_missing", "link_cross_country")},
    }


def metrics_bundle(ev: dict, recall_at_k: tuple[int, ...]) -> dict:
    return {
        "selected_bce": summarize(ev, STEPS[-1][1], ev["counts"][SELECTED], detailed=True, recall_at_k=recall_at_k),
        "phase1a_union_fallback": summarize(ev, STEPS[0][1], ev["counts"]["phase1a_union"]),
        "route_contributions": route_contributions(ev),
    }


def generate_tasks(ws: Workspace, algo: dict, ops: dict, target_path, active: list[int],
                   shard_positions: dict[int, dict[str, np.ndarray]], queries: dict, truth_targets: set[int],
                   invocation: dict, *, stop_after_tasks: int | None = None, only_sources: list[int] | None = None,
                   only_countries: list[str] | None = None, only_routes: list[str] | None = None) -> None:
    """Scan, fit, query, and checkpoint every pending (source, country, route, shard) task.

    Shared by fold evaluation and test inference so both run identical retrieval
    code. ``target_path(source)`` names the S2/S3 file to search. Raises
    StopRequested after ``stop_after_tasks`` newly completed tasks.
    """
    keys = sorted({key for groups in shard_positions.values() for key in groups})
    for source in algo["target_sources"]:
        if only_sources and source not in only_sources:
            continue
        path = target_path(source)
        for key in keys:
            if only_countries and key not in only_countries:
                continue
            slug = country_slug(key)
            pending = {route: [shard for shard in active if key in shard_positions[shard]
                               and not ws.task_done(source, slug, route, shard)]
                       if not only_routes or route in only_routes else [] for route in ROUTES}
            skipped = sum(1 for shard in active if key in shard_positions[shard]) * len(ROUTES) - sum(map(len, pending.values()))
            invocation["tasks_skipped"] += skipped
            if not any(pending.values()) and ws.partition_meta_path(source, slug).exists() \
                    and ws.truth_meta_path(source).exists():
                invocation["partitions_skipped"].append(f"S{source}/{key}")
                log(f"S{source}/{key}: all {skipped} tasks already complete; skipping scan")
                continue
            scan_started = time.perf_counter()
            with PeakRSS() as memory:
                targets = scan_target_partition(path, key, truth_targets)
            scan_seconds = time.perf_counter() - scan_started
            record_partition(ws, source, slug, key, targets, truth_targets)
            invocation["partitions_scanned"].append(f"S{source}/{key}")
            append_jsonl(ws.timing, {"invocation": invocation["invocation_id"], "kind": "scan", "source": source,
                                     "country_key": key, "seconds": scan_seconds, "peak_rss": memory.peak,
                                     "targets": len(targets["ids"])})
            log(f"S{source}/{key}: scanned {len(targets['ids']):,} targets in {scan_seconds:.0f}s")
            active_positions = np.concatenate([shard_positions[shard][key] for shard in active
                                               if key in shard_positions[shard]])
            for route in ROUTES:
                todo = pending[route]
                if not todo:
                    continue
                ensure_memory(ops, f"S{source}/{key}/{route} fit")
                fit_started = time.perf_counter()
                with PeakRSS() as memory:
                    index = fit_route(route, targets, queries, active_positions, algo)
                fit_seconds = time.perf_counter() - fit_started
                invocation["fits"].append(f"S{source}/{key}/{route}")
                append_jsonl(ws.timing, {"invocation": invocation["invocation_id"], "kind": "fit",
                                         "source": source, "country_key": key, "route": route,
                                         "seconds": fit_seconds, "peak_rss": memory.peak})
                log(f"S{source}/{key}/{route}: fitted in {fit_seconds:.0f}s (peak {memory.peak/2**30:.2f} GiB)")
                for shard in todo:
                    positions = shard_positions[shard][key]
                    ensure_memory(ops, f"S{source}/{key}/{route} shard {shard}")
                    query_started = time.perf_counter()
                    with PeakRSS() as memory:
                        arrays = query_route(route, index, targets, queries, positions, algo, ops)
                    query_seconds = time.perf_counter() - query_started
                    task_path = ws.task_path(source, slug, route, shard)
                    atomic_savez(task_path, arrays)
                    atomic_write_json(task_path.with_suffix(".json"), {
                        "source": source, "country_key": key, "route": route, "shard": shard,
                        "queries": len(positions), "candidates": int(arrays["counts"].sum()
                                                                   if route in DENSE_ROUTES else len(arrays["ids"])),
                        "query_seconds": query_seconds, "peak_rss": memory.peak,
                        "bytes": task_path.stat().st_size, "file_sha256": sha256_file(task_path),
                        "content_sha256": content_hash(arrays),
                        "invocation": invocation["invocation_id"],
                    })
                    invocation["tasks_completed"] += 1
                    log(f"S{source}/{key}/{route} shard {shard}: {len(positions):,} queries in "
                        f"{query_seconds:.1f}s ({len(positions)/max(query_seconds, 1e-9):.0f} q/s)")
                    if stop_after_tasks is not None and invocation["tasks_completed"] >= stop_after_tasks:
                        raise StopRequested
                del index
            del targets


def run(config: dict, root: Path, scope: str, *, stop_after_tasks: int | None = None,
        emit_union: bool = False, verify_hashes: bool = True, require_gate: bool = True,
        work_dir: Path | None = None, only_sources: list[int] | None = None,
        only_countries: list[str] | None = None, only_routes: list[str] | None = None) -> dict:
    """Generate or resume route shards for ``scope`` and evaluate them when complete.

    ``only_*`` filters restrict shard generation to disjoint partitions so several
    worker processes can share one work directory; filtered invocations never
    evaluate. A final unfiltered invocation skips completed tasks and evaluates.
    """
    filtered = bool(only_sources or only_countries or only_routes)
    if only_routes and not set(only_routes) <= set(ROUTES):
        raise ValueError(f"Unknown routes: {sorted(set(only_routes) - set(ROUTES))}")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    started = time.perf_counter()
    validate_config(config)
    algo, ops, paths = config["algorithm"], config["operational"], config["paths"]
    train_dir = root / paths["train_dir"]
    output = root / paths["output_dir"]
    ws = Workspace(work_dir or root / paths["work_dir"])
    if scope == "full" and require_gate:
        gate_files = [output / "pilot_reproduction.json", output / "benchmark_eta.json"]
        if not all(path.exists() for path in gate_files):
            raise RuntimeError("Full scope requires pilot_reproduction.json and benchmark_eta.json first")
        reproduction = json.loads(gate_files[0].read_text())
        if not reproduction.get("full_run_allowed", reproduction.get("all_equal")):
            raise RuntimeError("Pilot reproduction is neither exact nor within the platform tolerance; "
                               "refusing the full run")

    ordered = ordered_fold_ids(root / paths["folds"], algo["validation_fold"], algo["seed"])
    state = check_state(ws, config, ordered)
    size = algo["shard_size"]
    n_shards = (len(ordered) + size - 1) // size
    active = {"pilot": [0], "benchmark": list(range(min(ops["benchmark_shards"], n_shards))),
              "full": list(range(n_shards))}[scope]
    invocation = {"invocation_id": uuid.uuid4().hex[:12], "scope": scope,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
                  "stop_after_tasks": stop_after_tasks, "tasks_completed": 0, "tasks_skipped": 0,
                  "partitions_scanned": [], "partitions_skipped": [], "fits": [],
                  "filters": {"sources": only_sources, "countries": only_countries, "routes": only_routes}}
    log(f"{scope}: {len(ordered):,} fold S1 in {n_shards} shards; active shards {active[0]}..{active[-1]}")

    queries = load_queries(train_dir, ordered)
    truth_indptr, truth_values = load_truth(train_dir, ordered)
    truth_targets = set(truth_values.tolist())
    shard_positions: dict[int, dict[str, np.ndarray]] = {}
    for shard in active:
        grouped = defaultdict(list)
        for position in range(shard * size, min((shard + 1) * size, len(ordered))):
            grouped[queries["country_key"][position]].append(position)
        shard_positions[shard] = {key: np.asarray(values, dtype=np.int64) for key, values in grouped.items()}
    keys = sorted({key for groups in shard_positions.values() for key in groups})
    log(f"loaded queries and truth; dynamic country keys in scope: {keys}")

    try:
        generate_tasks(ws, algo, ops, lambda source: train_dir / f"train_source{source}.tsv", active,
                       shard_positions, queries, truth_targets, invocation, stop_after_tasks=stop_after_tasks,
                       only_sources=only_sources, only_countries=only_countries, only_routes=only_routes)
    except StopRequested:
        invocation.update(status="stopped", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          wall_seconds=time.perf_counter() - started, peak_rss=peak_rss_bytes())
        append_jsonl(ws.invocations, invocation)
        log(f"stopped after {invocation['tasks_completed']} newly completed tasks")
        return {"status": "stopped", "invocation": invocation}

    if filtered:
        invocation.update(status="generated", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                          wall_seconds=time.perf_counter() - started, peak_rss=peak_rss_bytes())
        append_jsonl(ws.invocations, invocation)
        log(f"filtered generation finished: {invocation['tasks_completed']} new tasks; run unfiltered to evaluate")
        return {"status": "generated", "invocation": invocation}

    eval_started = time.perf_counter()
    pool_by_key, target_meta = load_partition_metadata(ws, keys)
    with PeakRSS() as memory:
        ev = evaluate_shards(ws, active, shard_positions, queries, truth_indptr, truth_values, pool_by_key,
                             target_meta, ops["rrf_constant"], verify_hashes=verify_hashes, emit_union=emit_union)
    eval_seconds = time.perf_counter() - eval_started
    append_jsonl(ws.timing, {"invocation": invocation["invocation_id"], "kind": "evaluate",
                             "s1": len(ev["positions"]), "seconds": eval_seconds, "peak_rss": memory.peak})
    recall_at_k = tuple(ops["recall_at_k"])
    result = {
        "status": "complete", "scope": scope, "run_id": config["run_id"], "state": state,
        "shards": active, "n_shards_in_fold": n_shards, "fold_size": len(ordered),
        "retrieval_pool_sizes_by_country_key": pool_by_key,
        "metrics": metrics_bundle(ev, recall_at_k),
        "evaluation_seconds": eval_seconds, "evaluation_peak_rss": memory.peak,
    }
    if scope == "benchmark" and len(active) > 1:
        result["additional_shards_metrics"] = metrics_bundle(subset_eval(ev, ev["positions"] >= size), recall_at_k)
    invocation.update(status="complete", finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      wall_seconds=time.perf_counter() - started, peak_rss=peak_rss_bytes())
    append_jsonl(ws.invocations, invocation)
    result["invocation"] = invocation
    result["_ev"] = ev
    result["_ordered"] = ordered
    result["_queries"] = queries
    return result


def record_partition(ws: Workspace, source: int, slug: str, key: str, targets: dict, truth_targets: set[int]) -> None:
    """Persist partition size/hash and true-target metadata; fail on drift."""
    meta = {"source": source, "country_key": key, "targets": len(targets["ids"]),
            "target_ids_sha256": hashlib.sha256(targets["ids_np"].tobytes()).hexdigest()}
    path = ws.partition_meta_path(source, slug)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != meta:
            raise RuntimeError(f"Target partition drift for S{source}/{key}: {existing} != {meta}")
    else:
        atomic_write_json(path, meta)
    items = sorted(targets["meta"].items())
    arrays = {
        "ids": np.asarray([k for k, _ in items], dtype=np.int64),
        "non_ascii": np.asarray([v[0] for _, v in items], dtype=np.bool_),
        "address_missing": np.asarray([v[1] for _, v in items], dtype=np.bool_),
        "country_key": np.asarray([v[2] for _, v in items], dtype=np.str_),
    }
    truth_path = ws.truth_meta_path(source)
    digest = content_hash(arrays)
    if truth_path.exists():
        with np.load(truth_path, allow_pickle=False) as data:
            existing = {name: data[name] for name in data.files}
        if content_hash(existing) != digest:
            raise RuntimeError(f"True-target metadata drift for S{source}")
    else:
        atomic_savez(truth_path, arrays)


def load_partition_metadata(ws: Workspace, keys: list[str]) -> tuple[dict[str, int], dict[int, tuple[bool, bool, str]]]:
    pool_by_key = {}
    for key in keys:
        pool_by_key[key] = sum(json.loads(ws.partition_meta_path(source, country_slug(key)).read_text())["targets"]
                               for source in (2, 3))
    target_meta = {}
    for source in (2, 3):
        with np.load(ws.truth_meta_path(source), allow_pickle=False) as data:
            for value, non_ascii, missing, key in zip(data["ids"], data["non_ascii"],
                                                      data["address_missing"], data["country_key"]):
                target_meta[int(value)] = (bool(non_ascii), bool(missing), str(key))
    return pool_by_key, target_meta


# ---------------------------------------------------------------------------
# Pilot reproduction, ETA, reporting


PILOT_FIELDS = ("positive_link_recall", "retrieved_true_links", "positive_s1_with_every_true_match_pct",
                "recall_by_country", "non_ascii_recall", "missing_target_address_recall",
                "candidate_reduction_ratio")
COUNT_FIELDS = ("mean", "median", "p95", "p99", "max", "total")


def pilot_reproduction(config: dict, root: Path, result: dict, ws: Workspace) -> dict:
    """Compare shard 0 with the Phase 1B selected metrics and cached route candidates."""
    from .evaluate_blocking import selected_ids

    reference = config["pilot_reference"]
    algo = config["algorithm"]
    checks: dict[str, bool] = {}
    for relative, expected in reference["expected_sha256"].items():
        checks[f"reference_sha256/{relative}"] = sha256_file(root / relative) == expected
    pilot_ids = selected_ids(root / config["paths"]["folds"], algo["validation_fold"], reference["sample_size"], algo["seed"])
    ordered = result["_ordered"]
    checks["shard0_equals_pilot_ids_in_order"] = ordered[:reference["sample_size"]] == pilot_ids
    ev = result["_ev"]
    keep = ev["positions"] < reference["sample_size"]
    pilot_ev = subset_eval(ev, keep)
    phase1b = json.loads((root / reference["phase1b_metrics"]).read_text())
    comparisons = {}
    for label, mask_key, step in (("selected_bce", "22", SELECTED), ("phase1a_union", "0", "phase1a_union")):
        expected = phase1b["configurations"][mask_key]
        observed = summarize(pilot_ev, dict(STEPS)[step], pilot_ev["counts"][step])
        observed_counts = {key: observed["candidate_count"][key] for key in COUNT_FIELDS}
        comparisons[label] = {"expected": {field: expected[field] for field in PILOT_FIELDS} | {"candidate_count": {
            key: expected["candidate_count"][key] for key in COUNT_FIELDS}},
            "observed": {field: observed[field] for field in PILOT_FIELDS} | {"candidate_count": observed_counts}}
        for field in PILOT_FIELDS:
            checks[f"{label}/{field}"] = observed[field] == expected[field]
        for key in COUNT_FIELDS:
            checks[f"{label}/candidate_count/{key}"] = observed_counts[key] == expected["candidate_count"][key]
    caches = [root / reference["phase1a_candidates"], root / reference["phase1b_candidates"]]
    if all(path.exists() for path in caches):
        candidate_checks, score_diffs = compare_pilot_candidates(root, reference, result, pilot_ids, ws)
        checks.update({f"candidates/{key}": value == 0 for key, value in candidate_checks.items()})
    else:
        # Lightweight cloud copies may omit the large pilot caches; the metric-level
        # checks still run and shard 0 must then be compared against the caches locally.
        candidate_checks, score_diffs = None, None
    nonfloat = {
        key: shard_id_hash(ws.task_path(int(key.split("/")[0][1:]), country_slug(key.split("/")[1]),
                                        key.split("/")[2], 0)) == expected
        for key, expected in reference.get("nonfloat_shard0_id_sha256", {}).items()
    }
    tolerance = platform_tolerance(comparisons["selected_bce"], config["gate"].get("platform_tolerance", {}))
    all_equal = all(checks.values())
    structural = all(value for key, value in checks.items()
                     if key.startswith("reference_sha256/") or key == "shard0_equals_pilot_ids_in_order")
    full_run_allowed = all_equal or bool(
        config["gate"].get("accept_platform_float_tolerance") and structural and nonfloat
        and all(nonfloat.values()) and tolerance["within"]
    )
    return {
        "all_equal": all_equal,
        "full_run_allowed": full_run_allowed,
        "basis": "exact" if all_equal else ("platform_float_tolerance" if full_run_allowed else "blocked"),
        "nonfloat_routes_identical_to_m1": nonfloat,
        "platform_tolerance": tolerance,
        "candidate_level_verified": candidate_checks is not None,
        "checks": checks,
        "candidate_mismatched_s1_by_route": candidate_checks,
        "max_abs_score_difference_vs_phase1a_cache": score_diffs,
        "comparisons": comparisons,
        "note": "Equality is exact (Python ==) for every metric; candidate sets and dense-route order are compared per S1.",
    }


def shard_id_hash(path: Path) -> str | None:
    """SHA-256 of a shard's positions and candidate IDs only (scores excluded)."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in ("positions", "ids", "counts", "indptr"):
            if key in data.files:
                digest.update(np.ascontiguousarray(data[key]).tobytes())
    return digest.hexdigest()


def platform_tolerance(comparison: dict, limits: dict) -> dict:
    """Pilot deltas versus Phase 1B for CPU-architecture float differences.

    Near-tied TF-IDF scores can swap target IDs at the top-k boundary on another
    CPU architecture. This bounds the metric effect; it never relaxes the
    requirement that data, order, and non-float routes match exactly.
    """
    expected, observed = comparison["expected"], comparison["observed"]
    ec, oc = expected["candidate_count"], observed["candidate_count"]
    deltas = {
        "retrieved_true_links_abs_diff": abs(observed["retrieved_true_links"] - expected["retrieved_true_links"]),
        "positive_every_match_pp_abs_diff": abs(observed["positive_s1_with_every_true_match_pct"]
                                                - expected["positive_s1_with_every_true_match_pct"]),
        "candidate_total_rel_diff": abs(oc["total"] - ec["total"]) / ec["total"],
        "percentiles_equal": all(oc[key] == ec[key] for key in ("median", "p95", "p99", "max")),
    }
    within = bool(limits) and (
        deltas["retrieved_true_links_abs_diff"] <= limits["max_retrieved_link_diff"]
        and deltas["positive_every_match_pp_abs_diff"] <= limits["max_every_match_pp_diff"]
        and deltas["candidate_total_rel_diff"] <= limits["max_candidate_total_rel_diff"]
        and (deltas["percentiles_equal"] or not limits.get("require_equal_percentiles", True))
    )
    return {"deltas": deltas, "limits": limits, "within": within}


def compare_pilot_candidates(root: Path, reference: dict, result: dict, pilot_ids: list[str],
                             ws: Workspace) -> tuple[dict, dict]:
    """Per-S1 route candidate comparison against the preserved Phase 1A/1B caches."""
    with np.load(root / reference["phase1a_candidates"], allow_pickle=False) as data:
        p1a = {name: data[name] for name in ("sample_ids", "exact_indptr", "exact_values", "name_ids",
                                               "name_scores", "address_ids", "address_scores")}
    with np.load(root / reference["phase1b_candidates"], allow_pickle=False) as data:
        p1b = {name: data[name] for name in data.files}
    if list(p1a["sample_ids"]) != pilot_ids or list(p1b["sample_ids"]) != pilot_ids:
        raise RuntimeError("Pilot caches have a different sample order")
    queries = result["_queries"]
    mismatches = Counter({route: 0 for route in ROUTES})
    score_diff = defaultdict(float)
    shard_keys = defaultdict(list)
    for position in range(len(pilot_ids)):
        shard_keys[queries["country_key"][position]].append(position)
    for key, positions in shard_keys.items():
        slug = country_slug(key)
        loaded = {}
        for route in ROUTES:
            for source in (2, 3):
                with np.load(ws.task_path(source, slug, route, 0), allow_pickle=False) as data:
                    loaded[route, source] = {name: data[name] for name in data.files}
        for local, position in enumerate(positions):
            for route, cache_ids, cache_scores in (("name_char", p1a["name_ids"], p1a["name_scores"]),
                                                   ("address_char", p1a["address_ids"], p1a["address_scores"]),
                                                   ("name_word", p1b["word"], None)):
                bad = False
                for source in (2, 3):
                    arrays = loaded[route, source]
                    count = int(arrays["counts"][local])
                    cached = cache_ids[position, source - 2]
                    cached = cached[cached != 0]
                    if not np.array_equal(arrays["ids"][local, :count].astype(np.uint64), cached):
                        bad = True
                    elif cache_scores is not None and count:
                        diff = float(np.max(np.abs(arrays["scores"][local, :count]
                                                   - cache_scores[position, source - 2, :count])))
                        score_diff[route] = max(score_diff[route], diff)
                mismatches[route] += bad
            for route, indptr, values in (("exact_name", p1a["exact_indptr"], p1a["exact_values"]),
                                          ("rare_name", p1b["rare_indptr"], p1b["rare_values"]),
                                          ("suffix_exact", p1b["suffix_indptr"], p1b["suffix_values"])):
                observed = set()
                for source in (2, 3):
                    arrays = loaded[route, source]
                    left, right = arrays["indptr"][local:local + 2]
                    observed.update(int(v) for v in arrays["ids"][left:right])
                expected = {int(v) for v in values[indptr[position]:indptr[position + 1]]}
                mismatches[route] += observed != expected
    return dict(mismatches), dict(score_diff)


def estimate_full_run(ws: Workspace, queries: dict, n_shards: int, eval_seconds_per_s1: float,
                      headroom: float = 1.25) -> dict:
    """Project full-fold wall time and disk from measured scan/fit/query/evaluation costs."""
    timing = read_jsonl(ws.timing)
    fold_by_key = Counter(queries["country_key"])
    scans, fits = {}, {}
    for entry in timing:
        if entry["kind"] == "scan":
            scans[entry["source"], entry["country_key"]] = max(scans.get((entry["source"], entry["country_key"]), 0), entry["seconds"])
        elif entry["kind"] == "fit":
            key = (entry["source"], entry["country_key"], entry["route"])
            fits[key] = max(fits.get(key, 0), entry["seconds"])
    done_queries, done_seconds, done_bytes = Counter(), Counter(), Counter()
    sidecar_peaks = []
    for sidecar in ws.shards.rglob("*.json"):
        row = json.loads(sidecar.read_text())
        sidecar_peaks.append(row["peak_rss"])
        key = (row["source"], row["country_key"], row["route"])
        done_queries[key] += row["queries"]
        done_seconds[key] += row["query_seconds"]
        done_bytes[key] += row["bytes"]
    parts = {}
    total_seconds = sum(scans.values())
    total_bytes = 0
    for key in sorted(fits):
        rate = done_seconds[key] / done_queries[key]
        remaining = fold_by_key[key[1]] - done_queries[key]
        seconds = fits[key] + rate * remaining
        byte_rate = done_bytes[key] / done_queries[key]
        parts["/".join(map(str, key))] = {"fit_seconds": fits[key], "query_seconds_per_s1": rate,
                                          "remaining_queries": remaining, "projected_seconds": seconds,
                                          "bytes_per_s1": byte_rate, "projected_new_bytes": byte_rate * remaining}
        total_seconds += seconds
        total_bytes += byte_rate * remaining
    eval_seconds = eval_seconds_per_s1 * len(queries["country_key"])
    total_seconds += eval_seconds
    peak = max([entry.get("peak_rss", 0) for entry in timing] + sidecar_peaks, default=0)
    return {
        "fold_s1": len(queries["country_key"]), "fold_s1_by_country_key": dict(fold_by_key), "shards": n_shards,
        "components": parts, "scan_seconds": sum(scans.values()), "evaluation_seconds": eval_seconds,
        "projected_wall_seconds": total_seconds, "headroom_factor": headroom,
        "projected_wall_seconds_with_headroom": total_seconds * headroom,
        "projected_new_shard_bytes": total_bytes,
        "projected_new_shard_bytes_with_headroom": total_bytes * headroom,
        "current_work_dir_bytes": directory_bytes(ws.work),
        "observed_peak_stage_rss_bytes": peak,
        "free_disk_bytes": shutil.disk_usage(ws.work).free,
    }


def environment() -> dict:
    import psutil
    return {
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "memory_bytes": psutil.virtual_memory().total,
        "packages": {package: importlib.metadata.version(package) for package in PACKAGES},
    }


def code_state(root: Path) -> dict:
    package = Path(__file__).resolve().parent
    return {"git_commit": None, "note": "Repository has no commits; source files are identified by SHA-256.",
            "files": {str(path.relative_to(root)): sha256_file(path) for path in sorted(package.glob("*.py"))}}


def input_hashes(root: Path, config: dict, ws: Workspace) -> dict:
    """SHA-256 of every training input, cached by (size, mtime) in the work directory."""
    cache_path = ws.work / "input_hashes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    paths = [root / config["paths"]["train_dir"] / name for name in
             ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv")]
    paths.append(root / config["paths"]["folds"])
    result = {}
    for path in paths:
        stat = path.stat()
        relative = str(path.relative_to(root))
        entry = cache.get(relative)
        if not entry or entry["size"] != stat.st_size or entry["mtime"] != stat.st_mtime:
            entry = {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": sha256_file(path)}
            cache[relative] = entry
        result[relative] = entry["sha256"]
    atomic_write_json(cache_path, cache)
    return result


def stage_runtime(ws: Workspace) -> dict:
    """Aggregate recorded scan/fit/query/evaluation seconds by stage key."""
    totals = defaultdict(float)
    peaks = defaultdict(int)
    for entry in read_jsonl(ws.timing):
        if entry["kind"] == "scan":
            key = f"S{entry['source']}/{entry['country_key']}/scan"
        elif entry["kind"] == "fit":
            key = f"S{entry['source']}/{entry['country_key']}/{entry['route']}/fit"
        else:
            key = "evaluate"
        totals[key] += entry["seconds"]
        peaks[key] = max(peaks[key], entry.get("peak_rss", 0))
    for sidecar in ws.shards.rglob("*.json"):
        row = json.loads(sidecar.read_text())
        key = f"S{row['source']}/{row['country_key']}/{row['route']}/query"
        totals[key] += row["query_seconds"]
        peaks[key] = max(peaks[key], row["peak_rss"])
    by_route = defaultdict(float)
    for key, seconds in totals.items():
        parts = key.split("/")
        if len(parts) == 4:
            by_route[parts[2]] += seconds
    return {"by_stage_seconds": dict(sorted(totals.items())), "by_stage_peak_rss": dict(sorted(peaks.items())),
            "by_route_seconds": dict(by_route), "note": "Sums across all invocations, including resumed and repeated fits."}


def gate_decision(config: dict, selected: dict, pilot_slices: dict | None) -> dict:
    gate = config["gate"]
    recall = selected["positive_link_recall"]
    every = selected["positive_s1_with_every_true_match_pct"]
    p95 = selected["candidate_count"]["p95"]
    checks = {
        "link_recall_within_0.5pp": recall >= gate["pilot_link_recall"] - gate["max_link_recall_drop_pp"] / 100,
        "positive_every_match_within_1pp": every >= gate["pilot_positive_every_match_pct"] - gate["max_every_match_drop_pp"],
        "p95_candidates_at_most_800": p95 <= gate["max_p95_candidates"],
    }
    slices = {}
    if pilot_slices:
        for name, (observed, expected) in pilot_slices.items():
            if observed is None or expected is None:
                continue
            slices[name] = {"observed": observed, "pilot": expected, "delta_pp": 100 * (observed - expected),
                            "collapsed": 100 * (observed - expected) < -2.0}
        checks["no_slice_drop_over_2pp"] = not any(item["collapsed"] for item in slices.values())
    return {"checks": checks, "slices_vs_pilot": slices, "passed": all(checks.values())}


def fmt_pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.3f}%"


def make_report(metrics: dict, manifest: dict) -> str:
    sel = metrics["metrics"]["selected_bce"]
    fb = metrics["metrics"]["phase1a_union_fallback"]
    contrib = metrics["metrics"]["route_contributions"]
    gate = metrics["gate"]
    c = sel["candidate_count"]
    lines = [
        "# Phase 1C — full validation-fold blocker evaluation (fold 0, B+C+E)", "",
        f"Status: **{metrics['status']}**. Evaluated {sel['evaluated_s1']:,} fold-0 S1 records "
        f"({metrics['n_shards_in_fold']} shards of {manifest['config']['algorithm']['shard_size']:,}) against the complete training S2/S3 corpus, "
        "partitioned by dynamic normalized country and target source. No test files were read, no gold links were injected, "
        "and target IDs were never deduplicated by text.", "",
        f"Phase 2 gate: **{'PASS' if gate['passed'] else 'FAIL'}**.", "",
        "| Gate check | Result |", "|---|---|",
        *[f"| {name} | {'pass' if okay else 'FAIL'} |" for name, okay in gate["checks"].items()], "",
        "## Candidate quality", "",
        "| Metric | Full fold | Pilot (25k) |", "|---|---:|---:|",
        f"| Evaluated S1 | {sel['evaluated_s1']:,} | 25,000 |",
        f"| Positive S1 / singletons | {sel['positive_s1']:,} / {sel['singleton_s1']:,} | 23,600 / 1,400 |",
        f"| True links | {sel['true_links']:,} | 86,372 |",
        f"| Retrieved true links | {sel['retrieved_true_links']:,} | 84,465 |",
        f"| Link recall | {fmt_pct(sel['positive_link_recall'])} | 97.792% |",
        f"| Positive S1 with every match | {sel['positive_s1_with_every_true_match_pct']:.3f}% | 93.50% |",
        f"| All S1 every match (singletons count as satisfied) | {sel['all_s1_every_match_pct']:.3f}% | — |",
        f"| Zero-candidate rate | {fmt_pct(sel['zero_candidate_rate'])} | — |",
        f"| S2 / S3 recall | {fmt_pct(sel['recall_by_source']['S2'])} / {fmt_pct(sel['recall_by_source']['S3'])} | — |",
        *[f"| {country} recall | {fmt_pct(value)} | {metrics['pilot_reference_slices'].get(country + ' recall', '—')} |"
          for country, value in sel["recall_by_country"].items()],
        f"| Non-ASCII recall ({sel['non_ascii_true_links']:,} links) | {fmt_pct(sel['non_ascii_recall'])} | 93.767% |",
        f"| Missing-target-address recall ({sel['missing_target_address_true_links']:,} links) | {fmt_pct(sel['missing_target_address_recall'])} | 88.949% |",
        f"| Cross-country true links (unreachable by design) | {sel['cross_country_true_links']:,} | — |", "",
        "## Candidate volume", "",
        "| Mean | Median | p90 | p95 | p99 | Max | Total pairs | Reduction vs same-country corpus |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {c['mean']:.1f} | {c['median']:g} | {c['p90']:g} | {c['p95']:g} | {c['p99']:g} | {c['max']:,} | {c['total']:,} | {100*sel['candidate_reduction_ratio']:.5f}% |", "",
        "Pilot: mean 544.8, median 541, p95 677, p99 729, max 791, reduction 99.98983%.", "",
        "### By country", "",
        "| Country | S1 | Links | Recall | Positive every-match | Non-ASCII | Missing address | Median | p95 | p99 | Max | Zero-cand. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for country, row in sel["by_country"].items():
        cc = row["candidate_count"]
        lines.append(f"| {country} | {row['s1']:,} | {row['true_links']:,} | {fmt_pct(row['link_recall'])} | "
                     f"{row['positive_every_match_pct']:.2f}% | {fmt_pct(row['non_ascii_recall'])} | "
                     f"{fmt_pct(row['missing_target_address_recall'])} | {cc['median']:g} | {cc['p95']:g} | "
                     f"{cc['p99']:g} | {cc['max']:,} | {fmt_pct(row['zero_candidate_rate'])} |")
    lines += ["", "### By true match count", "",
              "| True matches | S1 | Links | Link recall | Every-match % | Mean candidates |", "|---|---:|---:|---:|---:|---:|"]
    for bucket, row in sel["by_match_count_bucket"].items():
        every = "n/a" if row["every_match_pct"] is None else f"{row['every_match_pct']:.2f}%"
        mean = "n/a" if row["candidate_mean"] is None else f"{row['candidate_mean']:.1f}"
        lines.append(f"| {bucket} | {row['s1']:,} | {row['true_links']:,} | {fmt_pct(row['link_recall'])} | "
                     f"{every} | {mean} |")
    lines += ["", "Bucket 0 is singletons: every-match is vacuously satisfied.", "",
              "## Route contributions", "",
              "| Route | Alone: link recall | Alone: mean candidates | Unique true links |", "|---|---:|---:|---:|"]
    for route in ROUTES:
        row = contrib["route_alone"][route]
        lines.append(f"| {route} | {fmt_pct(row['link_recall'])} | {row['candidate_mean']:.1f} | {contrib['unique_true_links'][route]:,} |")
    lines += ["", "| Cumulative step | Link recall | Candidates | Marginal true links | Marginal candidates |",
              "|---|---:|---:|---:|---:|"]
    for step, row in contrib["cumulative"].items():
        lines.append(f"| {step} | {fmt_pct(row['link_recall'])} | {row['candidate_total']:,} | "
                     f"{row['marginal_true_links']:,} | {row['marginal_candidates']:,} |")
    fc = fb["candidate_count"]
    lines += ["", f"Phase 1A fallback union on the same fold: recall {fmt_pct(fb['positive_link_recall'])}, "
              f"positive every-match {fb['positive_s1_with_every_true_match_pct']:.2f}%, median/p95/max {fc['median']:g}/{fc['p95']:g}/{fc['max']:,}.", "",
              "RRF-ordered recall@k over the six-route union (constant 60; diagnostic ordering only): " +
              ", ".join(f"@{k} {fmt_pct(v)}" for k, v in sel["recall_at_k_rrf"].items()) + ".", ""]
    res = manifest["resources"]
    lines += ["## Resources and reproducibility", "",
              f"- Elapsed wall clock, first invocation start to last finish: {res['elapsed_wall_seconds']/3600:.2f} h "
              f"(sum over invocations, counting parallel workers separately: {res['all_invocations_wall_seconds']/3600:.2f} h; "
              f"final evaluation invocation: {res['full_invocation_wall_seconds']/3600:.2f} h).",
              f"- Peak process RSS: {res['peak_rss_bytes']/2**30:.2f} GiB.",
              f"- End-to-end throughput: {res['s1_per_second_end_to_end']:.1f} S1/s over the elapsed wall clock.",
              f"- Route shard footprint: {res['work_dir_bytes']/2**30:.2f} GiB under `artifacts/phase1c/work/` (retained for Phase 2).",
              "- Runtime by route (seconds, fit + query, all invocations): " +
              ", ".join(f"{k} {v:,.0f}" for k, v in sorted(res["stage_runtime"]["by_route_seconds"].items())) + ".",
              f"- Resume evidence: {len(manifest['invocations'])} invocations recorded; see `run_manifest.json` → `invocations`.",
              "- Config `configs/phase1c_fold0.json`; environment, source hashes, input hashes, and output hashes are in `run_manifest.json`.",
              "- Pilot reproduction on shard 0: " + {"exact": "exact",
                  "platform_float_tolerance": "within the documented CPU-architecture float tolerance (non-float routes identical)",
                  "blocked": "FAILED"}[manifest["pilot_reproduction_basis"]] +
              " (`artifacts/phase1c/pilot_reproduction.json`).", ""]
    return "\n".join(lines) + "\n"


def pilot_slices(selected: dict) -> dict:
    """(observed, pilot) pairs for slice-collapse checks."""
    pilot = {"India recall": 0.9615384615384616, "US recall": 0.9891363707286628,
             "non_ascii_recall": 0.9376732837972718, "missing_target_address_recall": 0.8894858019953953}
    observed = {f"{country} recall": value for country, value in selected["recall_by_country"].items()}
    observed["non_ascii_recall"] = selected["non_ascii_recall"]
    observed["missing_target_address_recall"] = selected["missing_target_address_recall"]
    return {name: (observed.get(name), value) for name, value in pilot.items()}


def public(result: dict) -> dict:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True,
                        help="pilot = shard 0 only; benchmark = first benchmark_shards shards; full = entire fold")
    parser.add_argument("--stop-after-tasks", type=int, default=None,
                        help="exit cleanly after this many newly completed shard tasks (resume testing)")
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--emit-union", action="store_true",
                        help="also write per-shard union candidate files (IDs, route bits, RRF)")
    parser.add_argument("--skip-hash-verify", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--only-source", type=int, action="append", choices=(2, 3),
                        help="generate shards for this target source only (repeatable; no evaluation)")
    parser.add_argument("--only-country", action="append",
                        help="generate shards for this normalized country key only (repeatable; no evaluation)")
    parser.add_argument("--only-route", action="append", choices=ROUTES,
                        help="generate shards for this route only (repeatable; no evaluation)")
    parser.add_argument("--extra-training-fold", action="store_true",
                        help="full scope on a fold other than 0, used only as additional training data. The "
                             "fold-0 pilot gate does not apply (run the platform check separately); writes "
                             "fold<k>_metrics.json and fold<k>_manifest.json instead of the fold-0 report.")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    root = ROOT
    output = (args.output_dir or root / config["paths"]["output_dir"]).resolve()
    if args.output_dir:
        config["paths"]["output_dir"] = str(output.relative_to(root))
    extra = args.extra_training_fold
    if extra and (args.scope != "full" or config["algorithm"]["validation_fold"] == 0):
        raise SystemExit("--extra-training-fold requires --scope full and a config whose validation_fold is not 0")
    result = run(config, root, args.scope, stop_after_tasks=args.stop_after_tasks, emit_union=args.emit_union,
                 verify_hashes=not args.skip_hash_verify, work_dir=args.work_dir, require_gate=not extra,
                 only_sources=args.only_source,
                 only_countries=[normalize_text(value) for value in args.only_country] if args.only_country else None,
                 only_routes=args.only_route)
    if result["status"] != "complete":
        return
    ws = Workspace(args.work_dir or root / config["paths"]["work_dir"])
    selected = result["metrics"]["selected_bce"]
    if extra:
        fold = config["algorithm"]["validation_fold"]
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / f"fold{fold}_metrics.json", public(result))
        atomic_write_json(output / f"fold{fold}_manifest.json", {
            "run_id": config["run_id"], "status": "complete", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "purpose": "additional training fold (no pilot gate; platform verified by a separate fold-0 pilot check)",
            "state": json.loads(ws.state.read_text()), "environment": environment(), "code_state": code_state(root),
            "input_sha256": input_hashes(root, config, ws), "invocations": read_jsonl(ws.invocations),
            "shard_content_sha256": {str(path.relative_to(ws.work)): json.loads(path.read_text())["content_sha256"]
                                     for path in sorted(ws.shards.rglob("*.json"))}})
        log(f"fold {fold}: recall {fmt_pct(selected['positive_link_recall'])}, positive every-match "
            f"{selected['positive_s1_with_every_true_match_pct']:.3f}%, p95 {selected['candidate_count']['p95']:g}")
        return
    log(f"{args.scope}: recall {fmt_pct(selected['positive_link_recall'])}, positive every-match "
        f"{selected['positive_s1_with_every_true_match_pct']:.3f}%, median/p95/max "
        f"{selected['candidate_count']['median']:g}/{selected['candidate_count']['p95']:g}/{selected['candidate_count']['max']}")
    reproduction_path = output / "pilot_reproduction.json"
    reproduction = pilot_reproduction(config, root, result, ws) | {
        "scope_run": args.scope, "invocation": result["invocation"]["invocation_id"]}
    atomic_write_json(reproduction_path, reproduction)
    log(f"pilot reproduction all_equal={reproduction['all_equal']} "
        f"full_run_allowed={reproduction['full_run_allowed']} basis={reproduction['basis']}")
    ev = result["_ev"]
    eval_rate = result["evaluation_seconds"] / len(ev["positions"])
    common = {
        "config_path": str(config_path.relative_to(root)), "config_sha256": sha256_file(config_path),
        "environment": environment(), "code_state": code_state(root),
        "input_sha256": input_hashes(root, config, ws),
    }
    if args.scope == "pilot":
        atomic_write_json(output / "pilot_metrics.json", public(result) | common)
    elif args.scope == "benchmark":
        eta = estimate_full_run(ws, result["_queries"], result["n_shards_in_fold"], eval_rate)
        benchmark_invocations = [row for row in read_jsonl(ws.invocations) if row["scope"] == "benchmark"]
        eta["benchmark_invocations"] = benchmark_invocations
        atomic_write_json(output / "benchmark_metrics.json", public(result) | common)
        atomic_write_json(output / "benchmark_eta.json", eta)
        log(f"projected full-fold wall time {eta['projected_wall_seconds']/3600:.2f} h "
            f"({eta['projected_wall_seconds_with_headroom']/3600:.2f} h with headroom); "
            f"new shard bytes {eta['projected_new_shard_bytes_with_headroom']/2**30:.2f} GiB with headroom")
    else:
        invocations = read_jsonl(ws.invocations)
        runtime = stage_runtime(ws)
        stamps = [time.mktime(time.strptime(row[key], "%Y-%m-%dT%H:%M:%S"))
                  for row in invocations for key in ("started", "finished") if row.get(key)]
        elapsed = max(stamps) - min(stamps) if stamps else result["invocation"]["wall_seconds"]
        metrics = public(result) | {
            "gate": gate_decision(config, selected, pilot_slices(selected)),
            "pilot_reference_slices": {"India recall": "96.154%", "US recall": "98.914%"},
        }
        metrics["gate"]["checks"]["pilot_reproduced_exact_or_within_platform_tolerance"] = bool(
            reproduction["full_run_allowed"])
        metrics["gate"]["pilot_reproduction_basis"] = reproduction["basis"]
        metrics["gate"]["passed"] = all(metrics["gate"]["checks"].values())
        metrics_path = output / "fold0_metrics.json"
        atomic_write_json(metrics_path, metrics)
        manifest = {
            "run_id": config["run_id"], "status": "complete", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "scope": "full", "validation_fold": config["algorithm"]["validation_fold"], "seed": config["algorithm"]["seed"],
            "config": config, **common, "state": result["state"],
            "invocations": invocations,
            "resources": {
                "full_invocation_wall_seconds": result["invocation"]["wall_seconds"],
                "all_invocations_wall_seconds": sum(row.get("wall_seconds", 0) for row in invocations),
                "peak_rss_bytes": max(row.get("peak_rss", 0) for row in invocations),
                "elapsed_wall_seconds": elapsed,
                "s1_per_second_end_to_end": len(ev["positions"]) / elapsed,
                "work_dir_bytes": directory_bytes(ws.work), "stage_runtime": runtime,
                "free_disk_bytes_after": shutil.disk_usage(ws.work).free,
            },
            "pilot_reproduction_all_equal": bool(reproduction["all_equal"]),
            "pilot_reproduction_basis": reproduction["basis"],
            "retention_policy": config["retention_policy"],
            "gate_passed": metrics["gate"]["passed"],
        }
        report = make_report(metrics, manifest)
        (output / "fold0_report.md").write_text(report)
        manifest["outputs_sha256"] = {
            str(path.relative_to(root)): sha256_file(path)
            for path in (metrics_path, output / "fold0_report.md", reproduction_path)
        }
        manifest["shard_content_sha256"] = {
            str(path.relative_to(ws.work)): json.loads(path.read_text())["content_sha256"]
            for path in sorted(ws.shards.rglob("*.json"))
        }
        atomic_write_json(output / "run_manifest.json", manifest)
        print(report, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
