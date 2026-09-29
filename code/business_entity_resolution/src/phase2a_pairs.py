"""Phase 2A pair dataset and candidate-only features from the frozen Phase 1C benchmark.

Run from code/business_entity_resolution (resumable; completed chunks are skipped):

    python3 -m src.phase2a_pairs --config ../../configs/phase2a_benchmark.json --stage all

Inputs are read, never written: the Phase 1C route shards for benchmark shards
0-2 (fold-0 positions 0-74,999) and the training TSVs. Candidates are exactly the
frozen B+C+E union; no pair is added, removed, or relabelled by text.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import json
import os
import re
import time
from array import array
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .blocking import encode_id
from .evaluate_blocking import ROOT
from .evaluate_phase1c import (
    DENSE_ROUTES, ROUTES, SCORED_RAGGED_ROUTES, Workspace, atomic_savez, atomic_write_json,
    candidate_union, content_hash, country_slug, load_queries, load_truth, ordered_fold_ids,
    rank_route, sha256_file,
)
from .normalization import has_non_ascii, normalize_text
from .phase1b_routes import strip_legal_suffix
from .phase2a_env import ResourceGuard, ResourceStop, log

SCORED = ("name_char", "address_char", "name_word", "rare_name")
DIGITS = re.compile(r"\d+")
FEATURES = (
    "is_s3", *[f"route_{route}" for route in ROUTES], "n_routes",
    *[f"score_{route}" for route in SCORED], *[f"rank_{route}" for route in SCORED],
    "rrf_score", "rrf_rank", "rrf_rank_frac", "cand_count", "log_cand_count", "s1_exact_hits",
    "name_exact", "name_suffix_exact", "name_ratio", "name_token_sort", "name_token_set",
    "name_partial", "name_jaro_winkler", "name_len_ratio",
    "addr_exact", "addr_ratio", "addr_token_set", "addr_partial",
    "digits_exact", "digits_token_set", "s1_has_digits", "tgt_has_digits",
    "s1_addr_missing", "tgt_addr_missing", "tgt_name_missing", "non_ascii_any",
)
UNAVAILABLE_FIELDS = ("phone", "email", "website")
FEATURE_BATCH_ROWS = 50_000


def split_of(position: int, split: dict) -> str:
    for name in ("train", "validation", "holdout"):
        low, high = split[name]
        if low <= position < high:
            return name
    raise ValueError(f"position {position} outside the Phase 2A benchmark")


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


class Layout:
    def __init__(self, out: Path):
        self.out = out
        self.pairs = out / "pairs"
        self.texts = out / "texts"
        self.features = out / "features"
        self.qa = out / "qa"
        self.logs = out / "logs"

    def pair_chunk(self, chunk: int) -> Path:
        return self.pairs / f"chunk{chunk:03d}.npz"

    def feature_chunk(self, chunk: int) -> Path:
        return self.features / f"chunk{chunk:03d}.npz"


# ---------------------------------------------------------------------------
# Stage 1: pair dataset


def s1_pairs(position: int, local: int, loaded: dict, truth: np.ndarray, rrf_constant: int) -> dict:
    """One S1's unique candidate pairs with route evidence, ranks, scores, RRF, and labels."""
    ranked, per_route = [], {}
    for route in ROUTES:
        parts, ids_all, scores_all, ranks_all = [], [], [], []
        for source in (2, 3):
            arrays = loaded[route, source]
            if route in DENSE_ROUTES:
                count = int(arrays["counts"][local])
                ids, scores = arrays["ids"][local, :count], arrays["scores"][local, :count]
            else:
                left, right = arrays["indptr"][local:local + 2]
                ids = arrays["ids"][left:right]
                scores = arrays["scores"][left:right] if route in SCORED_RAGGED_ROUTES else None
            parts.append((ids, scores))
            ids_all.append(ids.astype(np.int64))
            if scores is not None:
                scores_all.append(scores)
                ranks_all.append(np.arange(1, len(ids) + 1, dtype=np.int16))
        ranked.append(rank_route(parts))
        per_route[route] = (np.concatenate(ids_all),
                            np.concatenate(scores_all) if scores_all else None,
                            np.concatenate(ranks_all) if ranks_all else None)
    uids, bits, rank, rrf = candidate_union(ranked, rrf_constant)
    m = len(uids)
    if len(np.unique(uids)) != m:
        raise AssertionError("duplicate candidate pair")
    result = {"cand": uids.astype(np.uint32), "route_bits": bits, "rrf": rrf.astype(np.float32),
              "rrf_rank": rank.astype(np.int32)}
    for route in SCORED:
        ids, scores, ranks = per_route[route]
        score_col = np.full(m, np.nan, dtype=np.float32)
        rank_col = np.zeros(m, dtype=np.int16)
        if len(ids):
            where = np.searchsorted(uids, ids)
            score_col[where] = scores
            rank_col[where] = ranks
        result[f"score_{route}"] = score_col
        result[f"rank_{route}"] = rank_col
    label = np.isin(uids, truth)
    retrieved = int(np.isin(truth, uids).sum())
    if int(label.sum()) != retrieved:
        raise AssertionError("retrieved truth links not all labelled positive")
    result["label"] = label
    result["exact_hits"] = len(per_route["exact_name"][0])
    result["retrieved_truth"] = retrieved
    return result


def build_pairs(config: dict, root: Path, layout: Layout, guard: ResourceGuard) -> dict:
    inputs = config["inputs"]
    p1c = json.loads((root / inputs["phase1c_config"]).read_text())
    ws = Workspace(root / inputs["phase1c_work_dir"])
    shard_size = p1c["algorithm"]["shard_size"]
    chunk = config["chunk_s1"]
    ordered = ordered_fold_ids(root / inputs["folds"], p1c["algorithm"]["validation_fold"], p1c["algorithm"]["seed"])
    state = json.loads(ws.state.read_text())
    import hashlib
    if hashlib.sha256("\n".join(ordered).encode()).hexdigest() != state["order_sha256"]:
        raise RuntimeError("Fold order differs from the Phase 1C work directory")
    n = shard_size * len(inputs["benchmark_shards"])
    ids = ordered[:n]
    queries = load_queries(root / inputs["train_dir"], ids)
    truth_indptr, truth_values = load_truth(root / inputs["train_dir"], ids)
    input_hashes = {}
    for shard in inputs["benchmark_shards"]:
        starts = range(shard * shard_size, (shard + 1) * shard_size, chunk)
        if all(layout.pair_chunk(start // chunk).exists() for start in starts):
            log(f"pairs: shard {shard} already complete")
            continue
        guard.check(f"pairs shard {shard} load")
        keys = sorted({queries["country_key"][p] for p in range(shard * shard_size, (shard + 1) * shard_size)})
        loaded_by_key, locator = {}, {}
        for key in keys:
            loaded = {}
            for route in ROUTES:
                for source in (2, 3):
                    path = ws.task_path(source, country_slug(key), route, shard)
                    sidecar = json.loads(path.with_suffix(".json").read_text())
                    digest = sha256_file(path)
                    if digest != sidecar["file_sha256"]:
                        raise RuntimeError(f"Phase 1C shard checksum mismatch: {path}")
                    input_hashes[str(path.relative_to(root))] = digest
                    with np.load(path, allow_pickle=False) as data:
                        loaded[route, source] = {name: data[name] for name in data.files}
            positions = loaded["exact_name", 2]["positions"]
            for arrays in loaded.values():
                if not np.array_equal(arrays["positions"], positions):
                    raise RuntimeError("route shards disagree on query order")
            loaded_by_key[key] = loaded
            for local, position in enumerate(positions):
                locator[int(position)] = (key, local)
        for start in starts:
            index = start // chunk
            if layout.pair_chunk(index).exists():
                continue
            guard.check(f"pairs chunk {index}")
            rows, s1 = [], {"positions": [], "truth_len": [], "cand_count": [], "retrieved_truth": [], "exact_hits": []}
            for position in range(start, start + chunk):
                key, local = locator[position]
                truth = truth_values[truth_indptr[position]:truth_indptr[position + 1]]
                row = s1_pairs(position, local, loaded_by_key[key], truth, config["rrf_constant"])
                row["s1_pos"] = np.full(len(row["cand"]), position, dtype=np.int32)
                rows.append(row)
                s1["positions"].append(position)
                s1["truth_len"].append(len(truth))
                s1["cand_count"].append(len(row["cand"]))
                s1["retrieved_truth"].append(row["retrieved_truth"])
                s1["exact_hits"].append(row["exact_hits"])
            arrays = {name: np.concatenate([row[name] for row in rows]) for name in
                      ("s1_pos", "cand", "label", "route_bits", "rrf", "rrf_rank",
                       *[f"score_{r}" for r in SCORED], *[f"rank_{r}" for r in SCORED])}
            arrays["cand_count"] = np.repeat(np.asarray(s1["cand_count"], np.int32), s1["cand_count"])
            arrays.update({f"s1_{name}": np.asarray(values, dtype=np.int32) for name, values in s1.items()})
            arrays["s1_country_key"] = np.asarray([queries["country_key"][p] for p in s1["positions"]], dtype=np.str_)
            arrays["s1_country"] = np.asarray([queries["country"][p] for p in s1["positions"]], dtype=np.str_)
            pair_keys = arrays["s1_pos"].astype(np.int64) << 32 | arrays["cand"].astype(np.int64)
            if len(np.unique(pair_keys)) != len(pair_keys):
                raise AssertionError("duplicate (S1, candidate) rows")
            atomic_savez(layout.pair_chunk(index), arrays)
            atomic_write_json(layout.pair_chunk(index).with_suffix(".json"), {
                "chunk": index, "s1": chunk, "pairs": int(len(arrays["cand"])), "positives": int(arrays["label"].sum()),
                "content_sha256": content_hash(arrays)})
        log(f"pairs: shard {shard} written")
        del loaded_by_key, locator
    manifest = layout.pairs / "input_hashes.json"
    if input_hashes:
        existing = json.loads(manifest.read_text()) if manifest.exists() else {}
        atomic_write_json(manifest, existing | input_hashes)
    return {"s1": n}


# ---------------------------------------------------------------------------
# Stage 2: disk-backed text stores


def build_texts(config: dict, root: Path, layout: Layout, guard: ResourceGuard) -> None:
    """Normalized S1 texts for the benchmark and all training S2/S3 targets.

    Target strings go to concatenated UTF-8 byte files with offsets, so feature
    computation reads them through memory maps instead of holding ~10M Python
    strings in RAM.
    """
    inputs = config["inputs"]
    train = root / inputs["train_dir"]
    done = layout.texts / "targets_meta.npz"
    if not (layout.texts / "s1.npz").exists():
        p1c = json.loads((root / inputs["phase1c_config"]).read_text())
        ordered = ordered_fold_ids(root / inputs["folds"], p1c["algorithm"]["validation_fold"], p1c["algorithm"]["seed"])
        ids = ordered[:p1c["algorithm"]["shard_size"] * len(inputs["benchmark_shards"])]
        position = {entity_id: i for i, entity_id in enumerate(ids)}
        raw_missing = np.zeros(len(ids), np.bool_)
        with (train / "train_source1.tsv").open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                i = position.get(row["entity_id"])
                if i is not None:
                    raw_missing[i] = not row["business_address"].strip()
        queries = load_queries(train, ids)
        atomic_savez(layout.texts / "s1.npz", {
            "entity_id": np.asarray(ids, dtype=np.str_), "name": np.asarray(queries["name"], dtype=np.str_),
            "address": np.asarray(queries["address"], dtype=np.str_), "non_ascii": queries["non_ascii"],
            "address_missing": raw_missing, "country": np.asarray(queries["country"], dtype=np.str_)})
        log("texts: S1 store written")
    if done.exists():
        log("texts: target store already complete")
        return
    layout.texts.mkdir(parents=True, exist_ok=True)
    ids = array("I")
    name_off, addr_off = array("q", [0]), array("q", [0])
    non_ascii, addr_missing, name_missing = bytearray(), bytearray(), bytearray()
    names_tmp = layout.texts / "targets_name.bin.tmp"
    addrs_tmp = layout.texts / "targets_address.bin.tmp"
    with names_tmp.open("wb") as names_file, addrs_tmp.open("wb") as addrs_file:
        for source in (2, 3):
            guard.check(f"texts S{source}")
            with (train / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file, delimiter="\t"):
                    name = normalize_text(row["business_name"]).encode()
                    address = normalize_text(row["business_address"]).encode()
                    ids.append(encode_id(row["entity_id"]))
                    names_file.write(name)
                    addrs_file.write(address)
                    name_off.append(name_off[-1] + len(name))
                    addr_off.append(addr_off[-1] + len(address))
                    non_ascii.append(has_non_ascii(row["business_name"]) or has_non_ascii(row["business_address"]))
                    addr_missing.append(not row["business_address"].strip())
                    name_missing.append(not row["business_name"].strip())
            log(f"texts: S{source} streamed ({len(ids):,} targets so far)")
    names_tmp.replace(layout.texts / "targets_name.bin")
    addrs_tmp.replace(layout.texts / "targets_address.bin")
    ids_np = np.frombuffer(ids, dtype=np.uint32)
    order = np.argsort(ids_np, kind="stable")
    atomic_savez(done, {
        "sorted_ids": ids_np[order], "order": order.astype(np.int64),
        "name_offsets": np.frombuffer(name_off, dtype=np.int64), "address_offsets": np.frombuffer(addr_off, dtype=np.int64),
        "non_ascii": np.frombuffer(bytes(non_ascii), dtype=np.bool_),
        "address_missing": np.frombuffer(bytes(addr_missing), dtype=np.bool_),
        "name_missing": np.frombuffer(bytes(name_missing), dtype=np.bool_)})
    log("texts: target store complete")


class TargetStore:
    """Read-only target text lookup; strings are fetched with pread so they stay in
    the reclaimable OS file cache instead of this process's resident memory."""

    def __init__(self, texts: Path):
        with np.load(texts / "targets_meta.npz", allow_pickle=False) as data:
            self.meta = {name: data[name] for name in data.files}
        for name in ("order", "name_offsets", "address_offsets"):
            values = self.meta[name]
            if values.max(initial=0) >= 2**31:
                raise OverflowError(f"{name} exceeds int32")
            self.meta[name] = values.astype(np.int32)
        self.files = {"name": os.open(texts / "targets_name.bin", os.O_RDONLY),
                      "address": os.open(texts / "targets_address.bin", os.O_RDONLY)}

    def rows(self, ids: np.ndarray) -> np.ndarray:
        where = np.searchsorted(self.meta["sorted_ids"], ids)
        if not np.array_equal(self.meta["sorted_ids"][np.minimum(where, len(self.meta["sorted_ids"]) - 1)], ids):
            raise KeyError("candidate ID missing from the target store")
        return self.meta["order"][where]

    def text(self, row: int, field: str) -> str:
        offsets = self.meta["name_offsets" if field == "name" else "address_offsets"]
        start, end = int(offsets[row]), int(offsets[row + 1])
        return os.pread(self.files[field], end - start, start).decode() if end > start else ""


# ---------------------------------------------------------------------------
# Stage 3: features


def sample_train_rows(pairs: dict, seed: int, chunk: int, hard: int, random_count: int) -> np.ndarray:
    """All positives, the top-RRF negatives, and seeded random negatives per S1."""
    rng = np.random.default_rng([seed, chunk])
    keep = []
    s1_pos, label, rrf_rank = pairs["s1_pos"], pairs["label"], pairs["rrf_rank"]
    boundaries = np.flatnonzero(np.diff(s1_pos)) + 1
    for rows in np.split(np.arange(len(s1_pos)), boundaries):
        positives = rows[label[rows]]
        negatives = rows[~label[rows]]
        negatives = negatives[np.argsort(rrf_rank[negatives], kind="stable")]
        top, rest = negatives[:hard], negatives[hard:]
        sampled = rng.choice(rest, size=min(random_count, len(rest)), replace=False) if len(rest) else rest
        keep.append(np.sort(np.concatenate([positives, top, sampled])))
    return np.concatenate(keep) if keep else np.zeros(0, np.int64)


def pairwise(scorer, left: list[str], right: list[str]) -> np.ndarray:
    return np.asarray(process.cpdist(left, right, scorer=scorer, workers=phase2a_env.WORKERS), dtype=np.float32)


def compute_features(pairs: dict, rows: np.ndarray, s1: dict, targets: TargetStore) -> np.ndarray:
    """Deterministic candidate-only features for selected pair rows (float32, NaN = not applicable)."""
    n = len(rows)
    columns: dict[str, np.ndarray] = {}
    cand = pairs["cand"][rows]
    bits = pairs["route_bits"][rows]
    s1_index = pairs["s1_pos"][rows]
    columns["is_s3"] = (cand & 1).astype(np.float32)
    for i, route in enumerate(ROUTES):
        columns[f"route_{route}"] = ((bits >> i) & 1).astype(np.float32)
    columns["n_routes"] = sum(columns[f"route_{route}"] for route in ROUTES)
    for route in SCORED:
        columns[f"score_{route}"] = pairs[f"score_{route}"][rows].astype(np.float32)
        rank = pairs[f"rank_{route}"][rows].astype(np.float32)
        columns[f"rank_{route}"] = np.where(rank > 0, rank, np.nan)
    columns["rrf_score"] = pairs["rrf"][rows]
    columns["rrf_rank"] = pairs["rrf_rank"][rows].astype(np.float32)
    count = pairs["cand_count"][rows].astype(np.float32)
    columns["rrf_rank_frac"] = columns["rrf_rank"] / np.maximum(count, 1)
    columns["cand_count"] = count
    columns["log_cand_count"] = np.log1p(count)
    s1_lookup = {int(p): i for i, p in enumerate(pairs["s1_positions"])}
    exact_hits = pairs["s1_exact_hits"]
    columns["s1_exact_hits"] = np.asarray([exact_hits[s1_lookup[int(p)]] for p in s1_index], dtype=np.float32)

    target_rows = targets.rows(cand.astype(np.uint32))
    q_name = [s1["name"][p] for p in s1_index]
    q_addr = [s1["address"][p] for p in s1_index]
    cache: dict[int, tuple[str, str]] = {}
    t_name, t_addr = [], []
    for row in target_rows:
        row = int(row)
        texts = cache.get(row)
        if texts is None:
            texts = cache[row] = (targets.text(row, "name"), targets.text(row, "address"))
        t_name.append(texts[0])
        t_addr.append(texts[1])
    q_digits = [" ".join(DIGITS.findall(value)) for value in q_addr]
    t_digits = [" ".join(DIGITS.findall(value)) for value in t_addr]

    name_both = np.asarray([bool(a) and bool(b) for a, b in zip(q_name, t_name)])
    addr_both = np.asarray([bool(a) and bool(b) for a, b in zip(q_addr, t_addr)])
    digit_both = np.asarray([bool(a) and bool(b) for a, b in zip(q_digits, t_digits)])
    columns["name_exact"] = np.asarray([a == b and bool(a) for a, b in zip(q_name, t_name)], dtype=np.float32)
    columns["name_suffix_exact"] = np.asarray(
        [bool(a) and strip_legal_suffix(a) == strip_legal_suffix(b) for a, b in zip(q_name, t_name)], dtype=np.float32)
    for column, scorer in (("name_ratio", fuzz.ratio), ("name_token_sort", fuzz.token_sort_ratio),
                           ("name_token_set", fuzz.token_set_ratio), ("name_partial", fuzz.partial_ratio)):
        columns[column] = np.where(name_both, pairwise(scorer, q_name, t_name) / 100, np.nan)
    columns["name_jaro_winkler"] = np.where(name_both, pairwise(JaroWinkler.normalized_similarity, q_name, t_name), np.nan)
    q_len = np.asarray([len(v) for v in q_name], dtype=np.float32)
    t_len = np.asarray([len(v) for v in t_name], dtype=np.float32)
    columns["name_len_ratio"] = np.where(name_both, np.minimum(q_len, t_len) / np.maximum(np.maximum(q_len, t_len), 1), np.nan)
    columns["addr_exact"] = np.where(addr_both, np.asarray([a == b for a, b in zip(q_addr, t_addr)], dtype=np.float32), np.nan)
    for column, scorer in (("addr_ratio", fuzz.ratio), ("addr_token_set", fuzz.token_set_ratio),
                           ("addr_partial", fuzz.partial_ratio)):
        columns[column] = np.where(addr_both, pairwise(scorer, q_addr, t_addr) / 100, np.nan)
    columns["digits_exact"] = np.where(digit_both, np.asarray([a == b for a, b in zip(q_digits, t_digits)], dtype=np.float32), np.nan)
    columns["digits_token_set"] = np.where(digit_both, pairwise(fuzz.token_set_ratio, q_digits, t_digits) / 100, np.nan)
    columns["s1_has_digits"] = np.asarray([bool(v) for v in q_digits], dtype=np.float32)
    columns["tgt_has_digits"] = np.asarray([bool(v) for v in t_digits], dtype=np.float32)
    columns["s1_addr_missing"] = s1["address_missing"][s1_index].astype(np.float32)
    columns["tgt_addr_missing"] = targets.meta["address_missing"][target_rows].astype(np.float32)
    columns["tgt_name_missing"] = targets.meta["name_missing"][target_rows].astype(np.float32)
    columns["non_ascii_any"] = (s1["non_ascii"][s1_index] | targets.meta["non_ascii"][target_rows]).astype(np.float32)
    matrix = np.column_stack([np.asarray(columns[name], dtype=np.float32) for name in FEATURES])
    if matrix.shape != (n, len(FEATURES)) or np.isinf(matrix).any():
        raise AssertionError("feature matrix shape or infinity check failed")
    return matrix


def build_features(config: dict, root: Path, layout: Layout, guard: ResourceGuard) -> None:
    targets = TargetStore(layout.texts)
    with np.load(layout.texts / "s1.npz", allow_pickle=False) as data:
        s1 = {name: data[name] for name in data.files}
    s1["name"] = s1["name"].tolist()
    s1["address"] = s1["address"].tolist()
    sampling = config["sampling"]
    chunks = sorted(int(p.stem[5:]) for p in layout.pairs.glob("chunk*.npz"))
    for index in chunks:
        path = layout.feature_chunk(index)
        if path.exists():
            continue
        guard.check(f"features chunk {index}")
        started = time.perf_counter()
        with np.load(layout.pair_chunk(index), allow_pickle=False) as data:
            pairs = {name: data[name] for name in data.files}
        split = split_of(int(pairs["s1_positions"][0]), config["split"])
        if split_of(int(pairs["s1_positions"][-1]), config["split"]) != split:
            raise AssertionError("chunk straddles a split boundary")
        rows = (sample_train_rows(pairs, sampling["seed"], index, sampling["hard_negatives_per_s1"],
                                  sampling["random_negatives_per_s1"]) if split == "train"
                else np.arange(len(pairs["cand"])))
        # Row-wise features, computed in bounded sub-batches to cap peak memory.
        matrix = np.concatenate([compute_features(pairs, rows[i:i + FEATURE_BATCH_ROWS], s1, targets)
                                 for i in range(0, len(rows), FEATURE_BATCH_ROWS)]) if len(rows) else \
            np.zeros((0, len(FEATURES)), np.float32)
        atomic_savez(path, {"rows": rows.astype(np.int64), "X": matrix,
                            "label": pairs["label"][rows], "s1_pos": pairs["s1_pos"][rows]})
        atomic_write_json(path.with_suffix(".json"), {
            "chunk": index, "split": split, "rows": int(len(rows)), "positives": int(pairs["label"][rows].sum()),
            "seconds": time.perf_counter() - started, "content_sha256": content_hash({"X": matrix, "rows": rows})})
        log(f"features: chunk {index} ({split}) {len(rows):,} rows in {time.perf_counter() - started:.0f}s")


# ---------------------------------------------------------------------------
# QA


def data_qa(config: dict, layout: Layout) -> dict:
    """Pair-level integrity, blocker ceiling by split, and feature range checks."""
    totals = {name: {"s1": 0, "pairs": 0, "positive_pairs": 0, "truth_links": 0, "retrieved_truth": 0,
                     "singletons": 0, "feature_rows": 0} for name in ("train", "validation", "holdout")}
    nan_rates = np.zeros(len(FEATURES))
    ranges = np.stack([np.full(len(FEATURES), np.inf), np.full(len(FEATURES), -np.inf)])
    feature_rows = 0
    for path in sorted(layout.pairs.glob("chunk*.npz")):
        with np.load(path, allow_pickle=False) as data:
            split = split_of(int(data["s1_positions"][0]), config["split"])
            t = totals[split]
            t["s1"] += len(data["s1_positions"])
            t["pairs"] += len(data["cand"])
            t["positive_pairs"] += int(data["label"].sum())
            t["truth_links"] += int(data["s1_truth_len"].sum())
            t["retrieved_truth"] += int(data["s1_retrieved_truth"].sum())
            t["singletons"] += int((data["s1_truth_len"] == 0).sum())
            if int(data["label"].sum()) != int(data["s1_retrieved_truth"].sum()):
                raise AssertionError(f"{path.name}: positive labels differ from retrieved truth")
        feature_path = layout.feature_chunk(int(path.stem[5:]))
        if feature_path.exists():
            with np.load(feature_path, allow_pickle=False) as data:
                X = data["X"]
                t["feature_rows"] += len(X)
                nan_rates += np.isnan(X).sum(axis=0)
                feature_rows += len(X)
                ranges[0] = np.fmin(ranges[0], np.nanmin(np.where(np.isnan(X), np.inf, X), axis=0))
                ranges[1] = np.fmax(ranges[1], np.nanmax(np.where(np.isnan(X), -np.inf, X), axis=0))
    for t in totals.values():
        t["blocker_recall_ceiling"] = t["retrieved_truth"] / t["truth_links"] if t["truth_links"] else None
        t["mean_candidates"] = t["pairs"] / t["s1"] if t["s1"] else None
    return {
        "splits": totals,
        "features": {name: {"nan_rate": nan_rates[i] / feature_rows if feature_rows else None,
                            "min": float(ranges[0, i]), "max": float(ranges[1, i])}
                     for i, name in enumerate(FEATURES)},
        "feature_rows": feature_rows,
        "constant_by_construction": {"country_consistency": "all pairs share the S1 country (dynamic exact-country blocking); not a feature"},
        "unavailable_fields": {field: "not present in the supplied data (entity_id, business_name, business_address, country only)"
                               for field in UNAVAILABLE_FIELDS},
        "checks": {"no_duplicate_pairs": True, "all_retrieved_truth_positive": True, "no_infinite_features": True},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("pairs", "texts", "features", "qa", "all"), required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    layout = Layout(ROOT / config["paths"]["output_dir"])
    resources = config["resources"]
    command = f"cd code/business_entity_resolution && python3 -m src.phase2a_pairs --config {args.config} --stage {args.stage}"
    guard = ResourceGuard(resources["min_available_memory_gib"], resources["max_swap_growth_gib"], command)
    started = time.perf_counter()
    try:
        if args.stage in ("pairs", "all"):
            build_pairs(config, ROOT, layout, guard)
        if args.stage in ("texts", "all"):
            build_texts(config, ROOT, layout, guard)
        if args.stage in ("features", "all"):
            build_features(config, ROOT, layout, guard)
        if args.stage in ("qa", "all"):
            qa = data_qa(config, layout) | {"resources": guard.summary(),
                                            "config_sha256": sha256_file(config_path)}
            atomic_write_json(layout.qa / "data_qa.json", qa)
            log("qa: " + json.dumps(qa["splits"]))
    except ResourceStop as stop:
        log(str(stop))
        raise SystemExit(3)
    finally:
        entry = {"stage": args.stage, "seconds": time.perf_counter() - started, **guard.summary()}
        layout.logs.mkdir(parents=True, exist_ok=True)
        with (layout.logs / "stage_runs.jsonl").open("a") as file:
            file.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
