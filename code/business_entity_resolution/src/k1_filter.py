"""K1 experiment: learned first-stage candidate filter on full fold 0, plus error-structure analysis.

Run from code/business_entity_resolution; every stage is resumable:

    python3 -m src.k1_filter --config ../../configs/k1_filter.json \
        --phase1c-work-dir <Phase 1C work dir containing state.json and shards/> \
        --work-dir /content/k1_work --output-dir /content/drive/MyDrive/ber_k1 --stage all

Stages: ``stores`` -> ``sample`` -> ``train`` -> ``score`` -> ``report``.

The frozen blocker's candidates (Phase 1C route shards, verified against the run
manifest's content hashes) are re-ranked per S1 by a small LightGBM that sees only
cheap evidence: which routes retrieved a candidate and at what rank/score, and
hashed overlaps of number tokens (leading zeros removed), name tokens, and
address words. Train-split S1 are scored out-of-fold, so their pruned lists look
like test-time lists. The report measures gold retention versus K (validation,
holdout, and a US->India transfer model) against the pre-registered gates. It also
reports the full-scale structure of hard negatives (target ownership) and noise
patterns of gold pairs. The ``score`` stage keeps the top ``keep_top`` candidates
per S1 in the work directory.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import hashlib
import json
import os
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .blocking import decode_id, encode_id
from .evaluate_blocking import ROOT
from .evaluate_phase1c import (
    ROUTES, Workspace, atomic_savez, atomic_write_json, content_hash, country_slug, load_queries, load_truth,
    ordered_fold_ids, sha256_file,
)
from .normalization import has_non_ascii, normalize_text
from .phase2a_analysis import lookup_owner, owner_map, pattern_audit
from .phase2a_env import ResourceGuard, ResourceStop, log
from .phase2a_pairs import SCORED, s1_pairs, sample_train_rows, split_of

DIGITS = re.compile(r"\d+")
SPLITS = ("train", "validation", "holdout")
CHEAP_FEATURES = (
    "is_s3", *[f"route_{route}" for route in ROUTES], "n_routes",
    *[f"score_{route}" for route in SCORED], *[f"rank_{route}" for route in SCORED],
    "rrf_score", "rrf_rank", "rrf_rank_frac", "cand_count", "s1_exact_hits",
    "digits_overlap", "digits_s1_n", "digits_tgt_n", "digits_jaccard", "digits_all_s1_in_tgt", "house_match",
    "name_overlap", "name_overlap_frac", "name_idf_frac",
    "addr_overlap", "addr_overlap_frac", "addr_idf_frac",
    "name_len_ratio", "addr_len_ratio", "tgt_addr_missing", "tgt_name_missing", "tgt_non_ascii",
)
C = {name: i for i, name in enumerate(CHEAP_FEATURES)}
TARGET_ARRAYS = ("ids", "digits", "name", "addr", "name_len", "addr_len", "addr_missing", "name_missing",
                 "non_ascii", "name_offsets", "addr_offsets")


# ---------------------------------------------------------------------------
# Token stores


def token_hash(token: str) -> int:
    value = zlib.crc32(token.encode())
    return value or 1


def record_tokens(name: str, address: str, limits: dict) -> tuple[list[int], list[int], list[int]]:
    """Distinct canonical number tokens (leading zeros removed), name tokens, and address words.

    Order of first appearance is kept (the first number is usually the house number),
    then each list is truncated to its configured width.
    """
    digits = list(dict.fromkeys(token.lstrip("0") or "0" for token in DIGITS.findall(address)))
    names = list(dict.fromkeys(token for token in name.split() if len(token) >= 2))
    words = list(dict.fromkeys(token for token in address.split() if len(token) >= 2 and not token.isdigit()))
    return ([token_hash(t) for t in digits[:limits["digits"]]], [token_hash(t) for t in names[:limits["name"]]],
            [token_hash(t) for t in words[:limits["address"]]])


def count_rows(path: Path) -> int:
    with path.open("rb") as file:
        return sum(1 for _ in file) - 1


def fit_idf(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    uniq, counts = np.unique(matrix[matrix != 0], return_counts=True)
    n = matrix.shape[0]
    return uniq.astype(np.uint32), (np.log((n + 1) / (counts + 1)) + 1).astype(np.float32), float(np.log(n + 1) + 1)


def idf_lookup(hashes: np.ndarray, uniq: np.ndarray, idf: np.ndarray, default: float) -> np.ndarray:
    flat = np.asarray(hashes).ravel()
    where = np.minimum(np.searchsorted(uniq, flat), max(len(uniq) - 1, 0))
    found = (uniq[where] == flat) if len(uniq) else np.zeros(len(flat), bool)
    out = np.where(found, idf[where] if len(uniq) else 0, default).astype(np.float32)
    out[flat == 0] = 0
    return out.reshape(np.shape(hashes))


def build_stores(config: dict, root: Path, work: Path, ctx: dict, guard: ResourceGuard, *,
                 targets_dir: Path | None = None, prefix: str = "train") -> None:
    """Token hashes, lengths, flags, and UTF-8 text files for every S2/S3 target and every S1 in ``ctx``.

    ``targets_dir``/``prefix`` select the searched corpus (training by default; the test corpus at
    inference). IDs whose string form does not round-trip through ``encode_id``/``decode_id`` are
    recorded in ``id_exceptions.json`` so output files can reproduce them exactly.
    """
    stores = work / "stores"
    done = stores / "DONE.json"
    if done.exists():
        return
    stores.mkdir(parents=True, exist_ok=True)
    limits = config["tokens"]
    train = targets_dir or root / config["inputs"]["train_dir"]
    capacity = sum(count_rows(train / f"{prefix}_source{source}.tsv") for source in (2, 3))
    exceptions: dict[str, str] = {}
    D, N, A = limits["digits"], limits["name"], limits["address"]
    t = {"ids": np.zeros(capacity, np.uint32), "digits": np.zeros((capacity, D), np.uint32),
         "name": np.zeros((capacity, N), np.uint32), "addr": np.zeros((capacity, A), np.uint32),
         "name_len": np.zeros(capacity, np.uint16), "addr_len": np.zeros(capacity, np.uint16),
         "addr_missing": np.zeros(capacity, bool), "name_missing": np.zeros(capacity, bool),
         "non_ascii": np.zeros(capacity, bool), "name_offsets": np.zeros(capacity + 1, np.int64),
         "addr_offsets": np.zeros(capacity + 1, np.int64)}
    i = 0
    with (stores / "targets_name.bin").open("wb") as names_file, (stores / "targets_addr.bin").open("wb") as addrs_file:
        for source in (2, 3):
            guard.check(f"stores S{source}")
            with (train / f"{prefix}_source{source}.tsv").open(encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file, delimiter="\t"):
                    if i >= capacity:
                        raise RuntimeError("More target rows than lines; embedded newlines?")
                    name = normalize_text(row["business_name"])
                    address = normalize_text(row["business_address"])
                    name_bytes, addr_bytes = name.encode(), address.encode()
                    names_file.write(name_bytes)
                    addrs_file.write(addr_bytes)
                    t["name_offsets"][i + 1] = t["name_offsets"][i] + len(name_bytes)
                    t["addr_offsets"][i + 1] = t["addr_offsets"][i] + len(addr_bytes)
                    t["ids"][i] = encode_id(row["entity_id"])
                    if decode_id(int(t["ids"][i])) != row["entity_id"]:
                        exceptions[str(int(t["ids"][i]))] = row["entity_id"]
                    digits, names, words = record_tokens(name, address, limits)
                    t["digits"][i, :len(digits)] = digits
                    t["name"][i, :len(names)] = names
                    t["addr"][i, :len(words)] = words
                    t["name_len"][i] = min(len(name), 65535)
                    t["addr_len"][i] = min(len(address), 65535)
                    t["addr_missing"][i] = not row["business_address"].strip()
                    t["name_missing"][i] = not row["business_name"].strip()
                    t["non_ascii"][i] = has_non_ascii(row["business_name"]) or has_non_ascii(row["business_address"])
                    i += 1
            log(f"stores: S{source} done ({i:,} targets)")
    for name, values in t.items():
        np.save(stores / f"targets_{name}.npy", values[:i + 1] if name.endswith("offsets") else values[:i])
    ids = t["ids"][:i]
    order = np.argsort(ids, kind="stable")
    np.save(stores / "targets_sorted_ids.npy", ids[order])
    np.save(stores / "targets_order.npy", order.astype(np.int32))
    idf_meta = {}
    for field in ("name", "addr"):
        uniq, idf, default = fit_idf(t[field][:i])
        np.save(stores / f"idf_{field}_uniq.npy", uniq)
        np.save(stores / f"idf_{field}_values.npy", idf)
        idf_meta[field] = default
    del t
    n1 = build_s1_store(config, stores, ctx)
    atomic_write_json(stores / "id_exceptions.json", exceptions)
    atomic_write_json(done, {"targets": int(i), "s1": n1, "idf_default": idf_meta, "limits": limits,
                             "corpus": f"{prefix}_source2/3", "id_exceptions": len(exceptions)})
    log(f"stores: complete ({i:,} targets, {n1:,} S1)")


def build_s1_store(config: dict, directory: Path, ctx: dict) -> int:
    """Token hashes and lengths for every S1 in ``ctx`` (position order), saved as s1_*.npy."""
    directory.mkdir(parents=True, exist_ok=True)
    limits = config["tokens"]
    D, N, A = limits["digits"], limits["name"], limits["address"]
    q = ctx["queries"]
    n1 = len(q["name"])
    s1 = {"digits": np.zeros((n1, D), np.uint32), "name": np.zeros((n1, N), np.uint32),
          "addr": np.zeros((n1, A), np.uint32), "name_len": np.zeros(n1, np.uint16), "addr_len": np.zeros(n1, np.uint16)}
    for p in range(n1):
        digits, names, words = record_tokens(q["name"][p], q["address"][p], limits)
        s1["digits"][p, :len(digits)] = digits
        s1["name"][p, :len(names)] = names
        s1["addr"][p, :len(words)] = words
        s1["name_len"][p] = min(len(q["name"][p]), 65535)
        s1["addr_len"][p] = min(len(q["address"][p]), 65535)
    for name, values in s1.items():
        np.save(directory / f"s1_{name}.npy", values)
    return n1


class Stores:
    def __init__(self, work: Path, s1_dir: Path | None = None):
        d = work / "stores"
        meta = json.loads((d / "DONE.json").read_text())
        self.t = {name: np.load(d / f"targets_{name}.npy", mmap_mode="r") for name in TARGET_ARRAYS}
        self.sorted_ids = np.load(d / "targets_sorted_ids.npy")
        self.order = np.load(d / "targets_order.npy")
        self.idf = {field: (np.load(d / f"idf_{field}_uniq.npy"), np.load(d / f"idf_{field}_values.npy"),
                            meta["idf_default"][field]) for field in ("name", "addr")}
        s1_root = s1_dir or d
        self.s1 = {name: np.load(s1_root / f"s1_{name}.npy") for name in ("digits", "name", "addr", "name_len", "addr_len")}
        self.s1_name_idf = idf_lookup(self.s1["name"], *self.idf["name"])
        self.s1_addr_idf = idf_lookup(self.s1["addr"], *self.idf["addr"])
        self.fd = {"name": os.open(d / "targets_name.bin", os.O_RDONLY), "addr": os.open(d / "targets_addr.bin", os.O_RDONLY)}

    def rows(self, cand: np.ndarray) -> np.ndarray:
        where = np.minimum(np.searchsorted(self.sorted_ids, cand), len(self.sorted_ids) - 1)
        if not np.array_equal(self.sorted_ids[where], cand):
            raise KeyError("candidate missing from target store")
        return self.order[where]

    def text(self, row: int, field: str) -> str:
        offsets = self.t["name_offsets" if field == "name" else "addr_offsets"]
        start, end = int(offsets[row]), int(offsets[row + 1])
        return os.pread(self.fd[field], end - start, start).decode() if end > start else ""


# ---------------------------------------------------------------------------
# Cheap features


def overlap(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: how many non-zero tokens of ``a`` also occur in ``b`` (and which)."""
    matched = ((a[:, :, None] == b[:, None, :]) & (a[:, :, None] != 0)).any(axis=2)
    return matched.sum(axis=1).astype(np.float32), matched


def ratio_or_nan(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-12), np.nan).astype(np.float32)


def cheap_features(pairs: dict, rows: np.ndarray, stores: Stores, chunk_start: int) -> np.ndarray:
    """Near-free per-pair evidence for the first-stage filter (float32, NaN = not applicable)."""
    col: dict[str, np.ndarray] = {}
    cand = pairs["cand"][rows]
    p = pairs["s1_pos"][rows].astype(np.int64)
    bits = pairs["route_bits"][rows]
    col["is_s3"] = (cand & 1).astype(np.float32)
    for i, route in enumerate(ROUTES):
        col[f"route_{route}"] = ((bits >> i) & 1).astype(np.float32)
    col["n_routes"] = sum(col[f"route_{route}"] for route in ROUTES)
    for route in SCORED:
        col[f"score_{route}"] = pairs[f"score_{route}"][rows].astype(np.float32)
        rank = pairs[f"rank_{route}"][rows].astype(np.float32)
        col[f"rank_{route}"] = np.where(rank > 0, rank, np.nan)
    count = pairs["cand_count"][rows].astype(np.float32)
    col["rrf_score"] = pairs["rrf"][rows].astype(np.float32)
    col["rrf_rank"] = pairs["rrf_rank"][rows].astype(np.float32)
    col["rrf_rank_frac"] = col["rrf_rank"] / np.maximum(count, 1)
    col["cand_count"] = count
    col["s1_exact_hits"] = pairs["s1_exact_hits"][p - chunk_start].astype(np.float32)
    trow = stores.rows(cand)
    s1d, td = stores.s1["digits"][p], np.asarray(stores.t["digits"][trow])
    col["digits_overlap"], matched = overlap(s1d, td)
    n1 = (s1d != 0).sum(axis=1).astype(np.float32)
    n2 = (td != 0).sum(axis=1).astype(np.float32)
    col["digits_s1_n"], col["digits_tgt_n"] = n1, n2
    col["digits_jaccard"] = ratio_or_nan(col["digits_overlap"], n1 + n2 - col["digits_overlap"])
    col["digits_all_s1_in_tgt"] = ((col["digits_overlap"] == n1) & (n1 > 0)).astype(np.float32)
    col["house_match"] = ((s1d[:, 0] != 0) & (s1d[:, 0] == td[:, 0])).astype(np.float32)
    for field, s1_idf in (("name", stores.s1_name_idf), ("addr", stores.s1_addr_idf)):
        a, b = stores.s1[field][p], np.asarray(stores.t[field][trow])
        shared, matched = overlap(a, b)
        n_a = (a != 0).sum(axis=1).astype(np.float32)
        weights = s1_idf[p]
        col[f"{field}_overlap"] = shared
        col[f"{field}_overlap_frac"] = ratio_or_nan(shared, n_a)
        col[f"{field}_idf_frac"] = ratio_or_nan((matched * weights).sum(axis=1), weights.sum(axis=1))
    for field in ("name", "addr"):
        la = stores.s1[f"{field}_len"][p].astype(np.float32)
        lb = np.asarray(stores.t[f"{field}_len"][trow]).astype(np.float32)
        col[f"{field}_len_ratio"] = ratio_or_nan(np.minimum(la, lb), np.maximum(la, lb))
    col["tgt_addr_missing"] = np.asarray(stores.t["addr_missing"][trow]).astype(np.float32)
    col["tgt_name_missing"] = np.asarray(stores.t["name_missing"][trow]).astype(np.float32)
    col["tgt_non_ascii"] = np.asarray(stores.t["non_ascii"][trow]).astype(np.float32)
    return np.column_stack([col[name] for name in CHEAP_FEATURES]).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1C shards and pair chunks


class Phase1CShards:
    """Read-only access to the Kaggle Phase 1C route shards, verified by content hash."""

    def __init__(self, work_dir: Path, manifest: dict):
        self.ws = Workspace(work_dir)
        self.expected = manifest["shard_content_sha256"]

    def load(self, shard: int, keys: list[str]) -> tuple[dict, dict]:
        loaded_by_key, locator = {}, {}
        for key in keys:
            slug = country_slug(key)
            loaded = {}
            for route in ROUTES:
                for source in (2, 3):
                    path = self.ws.task_path(source, slug, route, shard)
                    with np.load(path, allow_pickle=False) as data:
                        arrays = {name: data[name] for name in data.files}
                    rel = str(path.with_suffix(".json").relative_to(self.ws.work))
                    if content_hash(arrays) != self.expected.get(rel):
                        raise RuntimeError(f"Phase 1C shard content differs from the run manifest: {rel}")
                    loaded[route, source] = arrays
            positions = loaded["exact_name", 2]["positions"]
            for arrays in loaded.values():
                if not np.array_equal(arrays["positions"], positions):
                    raise RuntimeError("route shards disagree on query order")
            loaded_by_key[key] = loaded
            for local, position in enumerate(positions):
                locator[int(position)] = (key, local)
        return loaded_by_key, locator


def build_chunk(start: int, end: int, loaded_by_key: dict, locator: dict, ctx: dict, rrf_constant: int) -> dict:
    """All unique (S1, candidate) pairs for fold positions [start, end), with labels and route evidence."""
    truth_indptr, truth_values = ctx["truth"]
    rows = []
    per_s1 = defaultdict(list)
    for position in range(start, end):
        key, local = locator[position]
        truth = truth_values[truth_indptr[position]:truth_indptr[position + 1]]
        row = s1_pairs(position, local, loaded_by_key[key], truth, rrf_constant)
        row["s1_pos"] = np.full(len(row["cand"]), position, dtype=np.int32)
        rows.append(row)
        per_s1["positions"].append(position)
        per_s1["truth_len"].append(len(truth))
        per_s1["retrieved_truth"].append(row["retrieved_truth"])
        per_s1["exact_hits"].append(row["exact_hits"])
        per_s1["cand_count"].append(len(row["cand"]))
    arrays = {name: np.concatenate([row[name] for row in rows]) for name in
              ("s1_pos", "cand", "label", "route_bits", "rrf", "rrf_rank",
               *[f"score_{r}" for r in SCORED], *[f"rank_{r}" for r in SCORED])}
    arrays["cand_count"] = np.repeat(np.asarray(per_s1["cand_count"], np.int32), per_s1["cand_count"])
    arrays.update({f"s1_{name}": np.asarray(values, dtype=np.int64) for name, values in per_s1.items()})
    return arrays


def shard_keys(ctx: dict, shard: int, size: int) -> list[str]:
    n = len(ctx["ordered"])
    return sorted({ctx["queries"]["country_key"][p] for p in range(shard * size, min((shard + 1) * size, n))})


def rank_within_s1(s1_pos: np.ndarray, score: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """0-based rank of each pair within its S1 by (-score, candidate ID)."""
    order = np.lexsort((cand, -score, s1_pos))
    sorted_s1 = s1_pos[order]
    starts = np.flatnonzero(np.r_[True, sorted_s1[1:] != sorted_s1[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    rank_sorted = np.arange(len(order)) - np.repeat(starts, lengths)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = rank_sorted
    return rank


def oracle_f05(tp: np.ndarray, truth_len: np.ndarray) -> np.ndarray:
    """Entity F0.5 of a perfect matcher that sees ``tp`` of ``truth_len`` gold links (no false positives)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (1.25 * tp + 0.25 * (truth_len - tp)), 0.0)
    return np.where(truth_len == 0, 1.0, f)


def token_substitutions(a: str, b: str) -> list[tuple[str, str]]:
    ta, tb = set(a.split()), set(b.split())
    only_a = {t for t in ta - tb if not t.isdigit()}
    only_b = {t for t in tb - ta if not t.isdigit()}
    if 0 < len(only_a) <= 2 and 0 < len(only_b) <= 2:
        return [(x, y) for x in sorted(only_a) for y in sorted(only_b)]
    return []


# ---------------------------------------------------------------------------
# Stages


def load_context(config: dict, root: Path, phase1c_work: Path) -> dict:
    inputs = config["inputs"]
    p1c = json.loads((root / inputs["phase1c_config"]).read_text())
    manifest = json.loads((root / inputs["phase1c_manifest"]).read_text())
    ordered = ordered_fold_ids(root / inputs["folds"], p1c["algorithm"]["validation_fold"], p1c["algorithm"]["seed"])
    state = json.loads((phase1c_work / "state.json").read_text())
    order_sha = hashlib.sha256("\n".join(ordered).encode()).hexdigest()
    if state["order_sha256"] != order_sha or state["algorithm_sha256"] != manifest["state"]["algorithm_sha256"]:
        raise RuntimeError("Phase 1C work directory does not match the fold order or the frozen blocker")
    if config["split"]["holdout"][1] != len(ordered):
        raise RuntimeError("Split does not cover the whole fold")
    train_dir = root / inputs["train_dir"]
    queries = load_queries(train_dir, ordered)
    keys = sorted(set(queries["country_key"]))
    raw_by_key = {}
    for raw, key in zip(queries["country"], queries["country_key"]):
        raw_by_key.setdefault(key, raw)
    return {"p1c": p1c, "manifest": manifest, "ordered": ordered, "queries": queries,
            "truth": load_truth(train_dir, ordered), "keys": keys, "raw_by_key": raw_by_key,
            "country_code": np.asarray([keys.index(k) for k in queries["country_key"]], dtype=np.int8),
            "s1_num": np.asarray([int(value.split("-", 1)[1]) for value in ordered], dtype=np.int64)}


def stage_sample(config: dict, ctx: dict, stores: Stores, shards: Phase1CShards, work: Path, guard: ResourceGuard) -> None:
    out = work / "sample"
    out.mkdir(parents=True, exist_ok=True)
    size = ctx["p1c"]["algorithm"]["shard_size"]
    chunk = config["chunk_s1"]
    end = config["split"]["validation"][1]
    s = config["sampling"]
    q = ctx["queries"]
    for shard in range(0, (end - 1) // size + 1):
        starts = [st for st in range(shard * size, min((shard + 1) * size, end), chunk)]
        todo = [st for st in starts if not (out / f"chunk{st // chunk:03d}.npz").exists()]
        if not todo:
            continue
        loaded, locator = shards.load(shard, shard_keys(ctx, shard, size))
        for start in todo:
            guard.check(f"sample chunk {start // chunk}")
            arrays = build_chunk(start, start + chunk, loaded, locator, ctx, config["rrf_constant"])
            split = split_of(start, config["split"])
            rows = sample_train_rows(arrays, s["seed"], start // chunk, s["top_rrf_negatives"], s["random_negatives"])
            X = cheap_features(arrays, rows, stores, start)
            positions = arrays["s1_pos"][rows].astype(np.int64)
            fold = positions % config["oof_folds"] if split == "train" else np.full(len(rows), -1)
            substitutions = {"address": Counter(), "name": Counter()}
            if split == "train":
                for i in np.flatnonzero(arrays["label"]):
                    p, trow = int(arrays["s1_pos"][i]), int(stores.rows(arrays["cand"][i:i + 1])[0])
                    substitutions["address"].update(token_substitutions(q["address"][p], stores.text(trow, "addr")))
                    substitutions["name"].update(token_substitutions(q["name"][p], stores.text(trow, "name")))
            atomic_write_json(out / f"chunk{start // chunk:03d}.subst.json",
                              {field: [[a, b, n] for (a, b), n in counter.items()] for field, counter in substitutions.items()})
            atomic_savez(out / f"chunk{start // chunk:03d}.npz", {
                "X": X, "label": arrays["label"][rows], "s1_pos": positions, "fold": fold.astype(np.int8),
                "country": ctx["country_code"][positions]})
        log(f"sample: shard {shard} done")


def load_samples(work: Path, config: dict, split: str) -> dict:
    chunk = config["chunk_s1"]
    low, high = config["split"][split]
    parts = defaultdict(list)
    for index in range(low // chunk, high // chunk):
        with np.load(work / "sample" / f"chunk{index:03d}.npz", allow_pickle=False) as data:
            for name in data.files:
                parts[name].append(data[name])
    return {name: np.concatenate(values) for name, values in parts.items()}


def model_names(config: dict) -> list[str]:
    return [f"fold{f}" for f in range(config["oof_folds"])] + ["final", "transfer"]


def stage_train(config: dict, ctx: dict, work: Path, output: Path, guard: ResourceGuard) -> None:
    import lightgbm as lgb

    models = output / "models"
    models.mkdir(parents=True, exist_ok=True)
    names = model_names(config)
    if all((models / f"filter_{name}.txt").exists() for name in names):
        return
    train = load_samples(work, config, "train")
    valid = load_samples(work, config, "validation")
    params = dict(config["lightgbm"])
    rounds, stopping = params.pop("num_boost_round"), params.pop("early_stopping_rounds")
    transfer = config["gates"]["transfer"]
    us = ctx["keys"].index(transfer["train_country_key"]) if transfer["train_country_key"] in ctx["keys"] else None
    masks = {f"fold{f}": train["fold"] != f for f in range(config["oof_folds"])}
    masks["final"] = np.ones(len(train["label"]), bool)
    if us is not None:
        masks["transfer"] = train["country"] == us
    record_path = models / "training.json"
    records = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name in names:
        path = models / f"filter_{name}.txt"
        if path.exists() or name not in masks:
            continue
        guard.check(f"train {name}")
        started = time.perf_counter()
        mask = masks[name]
        train_set = lgb.Dataset(train["X"][mask], train["label"][mask], feature_name=list(CHEAP_FEATURES), free_raw_data=True)
        valid_set = lgb.Dataset(valid["X"], valid["label"], reference=train_set)
        booster = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=[valid_set],
                            callbacks=[lgb.early_stopping(stopping, verbose=False)])
        temporary = path.with_name(path.name + ".tmp")
        booster.save_model(str(temporary), num_iteration=booster.best_iteration)
        os.replace(temporary, path)
        records[name] = {"rows": int(mask.sum()), "positives": int(train["label"][mask].sum()),
                         "best_iteration": int(booster.best_iteration), "seconds": time.perf_counter() - started,
                         "best_valid_logloss": float(booster.best_score["valid_0"]["binary_logloss"]),
                         "gain_top": dict(sorted(zip(CHEAP_FEATURES, map(float, booster.feature_importance("gain"))),
                                                 key=lambda kv: -kv[1])[:15])}
        atomic_write_json(record_path, records)
        log(f"train: filter_{name} best_iteration={booster.best_iteration} in {time.perf_counter() - started:.0f}s")
        del train_set, valid_set, booster


def new_split_stats(k_grid: list[int]) -> dict:
    return {"s1": 0, "positive_s1": 0, "truth_links": 0, "retrieved_gold": 0, "candidates": 0,
            "oracle_f05_full_sum": 0.0,
            "kept_gold": {str(k): 0 for k in k_grid}, "kept_pairs": {str(k): 0 for k in k_grid},
            "oracle_f05_sum": {str(k): 0.0 for k in k_grid}, "positive_s1_all_kept": {str(k): 0 for k in k_grid}}


def accumulate(stats: dict, rank: np.ndarray, label: np.ndarray, local: np.ndarray, m: int,
               truth_len: np.ndarray, retrieved: np.ndarray, cand_count: np.ndarray, k_grid: list[int]) -> None:
    stats["s1"] += m
    stats["positive_s1"] += int((truth_len > 0).sum())
    stats["truth_links"] += int(truth_len.sum())
    stats["retrieved_gold"] += int(retrieved.sum())
    stats["candidates"] += int(cand_count.sum())
    stats["oracle_f05_full_sum"] += float(oracle_f05(retrieved, truth_len).sum())
    for k in k_grid:
        kept = np.bincount(local[label & (rank < k)], minlength=m)
        stats["kept_gold"][str(k)] += int(kept.sum())
        stats["kept_pairs"][str(k)] += int(np.minimum(cand_count, k).sum())
        stats["oracle_f05_sum"][str(k)] += float(oracle_f05(kept, truth_len).sum())
        stats["positive_s1_all_kept"][str(k)] += int(((kept == retrieved) & (truth_len > 0)).sum())


def stage_score(config: dict, ctx: dict, stores: Stores, shards: Phase1CShards, work: Path, output: Path,
                guard: ResourceGuard) -> None:
    import lightgbm as lgb

    names = model_names(config)
    models = {name: lgb.Booster(model_file=str(output / "models" / f"filter_{name}.txt"))
              for name in names if (output / "models" / f"filter_{name}.txt").exists()}
    threads = config["lightgbm"]["num_threads"]
    k_grid = config["k_grid"]
    keep = config["keep_top"]
    size = ctx["p1c"]["algorithm"]["shard_size"]
    chunk = config["chunk_s1"]
    n = len(ctx["ordered"])
    split = config["split"]
    folds = config["oof_folds"]
    transfer = config["gates"]["transfer"]
    eval_code = ctx["keys"].index(transfer["eval_country_key"]) if transfer["eval_country_key"] in ctx["keys"] else -1
    stats_dir = output / "score_stats"
    topk_dir = work / "topk"
    stats_dir.mkdir(parents=True, exist_ok=True)
    topk_dir.mkdir(parents=True, exist_ok=True)
    sorted_targets = owners = None  # loaded lazily, only if a shard still needs scoring
    q = ctx["queries"]
    for shard in range((n + size - 1) // size):
        stats_path = stats_dir / f"shard{shard:03d}.json"
        if stats_path.exists():
            continue
        if sorted_targets is None:
            sorted_targets, owners = owner_map(ctx["train_dir"])
        started = time.perf_counter()
        loaded, locator = shards.load(shard, shard_keys(ctx, shard, size))
        low, high = shard * size, min((shard + 1) * size, n)
        kept_ids = np.zeros((high - low, keep), np.uint32)
        kept_scores = np.zeros((high - low, keep), np.float16)
        kept_counts = np.zeros(high - low, np.int16)
        stats = {"splits": {}, "transfer": new_split_stats(k_grid), "ownership": Counter(), "patterns": Counter(),
                 "scripts": Counter(), "name_bins": Counter(), "dropped_gold_examples": []}
        for start in range(low, high, chunk):
            guard.check(f"score chunk {start // chunk}")
            stop = min(start + chunk, high)
            arrays = build_chunk(start, stop, loaded, locator, ctx, config["rrf_constant"])
            rows = np.arange(len(arrays["cand"]))
            X = cheap_features(arrays, rows, stores, start)
            pos = arrays["s1_pos"].astype(np.int64)
            score = np.empty(len(pos), dtype=np.float64)
            is_train = pos < split["train"][1]
            for f in range(folds):
                mask = is_train & (pos % folds == f)
                if mask.any():
                    score[mask] = models[f"fold{f}"].predict(X[mask], num_threads=threads)
            if (~is_train).any():
                score[~is_train] = models["final"].predict(X[~is_train], num_threads=threads)
            label = arrays["label"]
            rank = rank_within_s1(pos, score, arrays["cand"].astype(np.int64))
            local = (pos - start).astype(np.int64)
            m = stop - start
            top = rank < keep
            kept_ids[pos[top] - low, rank[top]] = arrays["cand"][top]
            kept_scores[pos[top] - low, rank[top]] = score[top].astype(np.float16)
            kept_counts[start - low:stop - low] = np.minimum(arrays["s1_cand_count"], keep)
            s1_split = np.asarray([split_of(int(p), split) for p in arrays["s1_positions"]])
            s1_country = ctx["country_code"][arrays["s1_positions"]]
            for name in SPLITS:
                for code, key in enumerate(ctx["keys"]):
                    sel = (s1_split == name) & (s1_country == code)
                    if not sel.any():
                        continue
                    idx = np.flatnonzero(sel)
                    row_sel = np.isin(local, idx)
                    remap = np.full(m, -1)
                    remap[idx] = np.arange(len(idx))
                    target = stats["splits"].setdefault(name, {}).setdefault(ctx["raw_by_key"][key], new_split_stats(k_grid))
                    accumulate(target, rank[row_sel], label[row_sel], remap[local[row_sel]], len(idx),
                               arrays["s1_truth_len"][idx], arrays["s1_retrieved_truth"][idx],
                               arrays["s1_cand_count"][idx], k_grid)
            if "transfer" in models:
                sel = (s1_split == "validation") & (s1_country == eval_code)
                if sel.any():
                    idx = np.flatnonzero(sel)
                    row_sel = np.isin(local, idx)
                    remap = np.full(m, -1)
                    remap[idx] = np.arange(len(idx))
                    t_score = models["transfer"].predict(X[row_sel], num_threads=threads)
                    t_rank = rank_within_s1(pos[row_sel], t_score, arrays["cand"][row_sel].astype(np.int64))
                    accumulate(stats["transfer"], t_rank, label[row_sel], remap[local[row_sel]], len(idx),
                               arrays["s1_truth_len"][idx], arrays["s1_retrieved_truth"][idx],
                               arrays["s1_cand_count"][idx], k_grid)
            evaluation = ~is_train
            if evaluation.any():
                owner = lookup_owner(sorted_targets, owners, arrays["cand"][evaluation])
                this = ctx["s1_num"][pos[evaluation]]
                lab = label[evaluation]
                kind = np.where(lab, "positive", np.where(owner < 0, "neg_unowned", "neg_owned_by_other_s1"))
                Xe = X[evaluation]
                same_address = (Xe[:, C["digits_all_s1_in_tgt"]] == 1) & (np.nan_to_num(Xe[:, C["addr_overlap_frac"]]) >= 0.8)
                name_idf = np.nan_to_num(Xe[:, C["name_idf_frac"]])
                strata = {"all": np.ones(len(lab), bool), "same_address": same_address,
                          "strong_name": name_idf >= 0.8, "same_address_strong_name": same_address & (name_idf >= 0.8),
                          "same_address_weak_name": same_address & (name_idf < 0.3),
                          "filter_top10": rank[evaluation] < 10}
                if np.any(lab & (owner != this)):
                    raise AssertionError("a gold pair is not owned by its S1")
                for stratum, mask in strata.items():
                    for k in ("positive", "neg_unowned", "neg_owned_by_other_s1"):
                        stats["ownership"][f"{stratum}|{k}"] += int(np.sum(mask & (kind == k)))
                gold_rows = np.flatnonzero(evaluation & label)
                for i in gold_rows:
                    p, trow = int(pos[i]), int(stores.rows(arrays["cand"][i:i + 1])[0])
                    t_name, t_addr = stores.text(trow, "name"), stores.text(trow, "addr")
                    audit = pattern_audit(q["name"][p], q["address"][p], t_name, t_addr)
                    for key, value in audit.items():
                        if isinstance(value, bool) and value:
                            stats["patterns"][key] += 1
                    stats["patterns"]["gold_pairs"] += 1
                    stats["scripts"][audit["target_name_script"]] += 1
                    ts = audit["name_token_set"]
                    stats["name_bins"]["<50" if ts < 50 else "50-89" if ts < 90 else ">=90"] += 1
                    if rank[i] >= 60 and len(stats["dropped_gold_examples"]) < 12:
                        stats["dropped_gold_examples"].append({
                            "rank": int(rank[i]), "s1_name": q["name"][p], "s1_address": q["address"][p],
                            "cand_name": t_name, "cand_address": t_addr,
                            "digits_overlap": float(X[i, C["digits_overlap"]]),
                            "name_idf_frac": float(np.nan_to_num(X[i, C["name_idf_frac"]])),
                            "addr_idf_frac": float(np.nan_to_num(X[i, C["addr_idf_frac"]]))})
        atomic_savez(topk_dir / f"shard{shard:03d}.npz", {
            "positions": np.arange(low, high, dtype=np.int64), "ids": kept_ids, "scores": kept_scores,
            "counts": kept_counts})
        stats["ownership"] = dict(stats["ownership"])
        stats["patterns"] = dict(stats["patterns"])
        stats["scripts"] = dict(stats["scripts"])
        stats["name_bins"] = dict(stats["name_bins"])
        stats["seconds"] = time.perf_counter() - started
        atomic_write_json(stats_path, stats)
        log(f"score: shard {shard} done in {stats['seconds']:.0f}s")


def merge_stats(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict):
            merge_stats(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def curve(stats: dict, k_grid: list[int]) -> dict:
    retrieved = stats["retrieved_gold"]
    return {str(k): {
        "retention": stats["kept_gold"][str(k)] / retrieved if retrieved else None,
        "mean_kept_candidates": stats["kept_pairs"][str(k)] / stats["s1"] if stats["s1"] else None,
        "oracle_macro_f05": stats["oracle_f05_sum"][str(k)] / stats["s1"] if stats["s1"] else None,
        "positive_s1_all_gold_kept": stats["positive_s1_all_kept"][str(k)] / stats["positive_s1"] if stats["positive_s1"] else None,
    } for k in k_grid}


def stage_report(config: dict, ctx: dict, work: Path, output: Path) -> dict:
    k_grid = config["k_grid"]
    totals: dict = {}
    ownership, patterns, scripts, name_bins = Counter(), Counter(), Counter(), Counter()
    examples = []
    seconds = 0.0
    n_shards = (len(ctx["ordered"]) + ctx["p1c"]["algorithm"]["shard_size"] - 1) // ctx["p1c"]["algorithm"]["shard_size"]
    for shard in range(n_shards):
        stats = json.loads((output / "score_stats" / f"shard{shard:03d}.json").read_text())
        merge_stats(totals.setdefault("splits", {}), stats["splits"])
        merge_stats(totals.setdefault("transfer", {}), stats["transfer"])
        ownership.update(stats["ownership"]); patterns.update(stats["patterns"])
        scripts.update(stats["scripts"]); name_bins.update(stats["name_bins"])
        examples.extend(stats["dropped_gold_examples"])
        seconds += stats["seconds"]
    by_split = {}
    for name, per_country in totals["splits"].items():
        combined = {}
        for country_stats in per_country.values():
            merge_stats(combined, country_stats)
        by_split[name] = {"all": {"s1": combined["s1"], "retrieved_gold": combined["retrieved_gold"],
                                  "mean_candidates": combined["candidates"] / combined["s1"],
                                  "oracle_macro_f05_no_filter": combined["oracle_f05_full_sum"] / combined["s1"],
                                  "curve": curve(combined, k_grid)},
                          **{country: {"s1": s["s1"], "curve": curve(s, k_grid),
                                       "oracle_macro_f05_no_filter": s["oracle_f05_full_sum"] / s["s1"]}
                             for country, s in sorted(per_country.items())}}
    gates = config["gates"]
    target = gates["retention_target"]
    val_curve = by_split["validation"]["all"]["curve"]
    chosen = next((k for k in k_grid if k <= gates["max_k"] and val_curve[str(k)]["retention"] >= target), None)
    transfer_curve = curve(totals["transfer"], k_grid) if totals["transfer"].get("s1") else None
    decision = {"retention_target": target, "chosen_k": chosen}
    if chosen is not None:
        key = str(chosen)
        hold = by_split["holdout"]["all"]["curve"][key]
        decision.update(
            validation_retention=val_curve[key]["retention"], holdout_retention=hold["retention"],
            transfer_retention=transfer_curve[key]["retention"] if transfer_curve else None,
            validation_oracle_f05_at_k=val_curve[key]["oracle_macro_f05"],
            validation_oracle_f05_no_filter=by_split["validation"]["all"]["oracle_macro_f05_no_filter"],
            mean_kept_candidates=val_curve[key]["mean_kept_candidates"],
            projected_test_pairs=val_curve[key]["mean_kept_candidates"] * config["test_s1_count"],
            projected_test_pairs_no_filter=by_split["validation"]["all"]["mean_candidates"] * config["test_s1_count"])
        decision["gates"] = {
            "validation_retention": val_curve[key]["retention"] >= target,
            "holdout_retention": hold["retention"] >= target,
            "transfer_retention": (transfer_curve[key]["retention"] >= target) if transfer_curve else None,
            "k_at_most_max": chosen <= gates["max_k"]}
        decision["passed_k1_gates"] = all(v for v in decision["gates"].values() if v is not None)
        decision["pending_gate"] = "end-to-end matcher F0.5 with vs without filter within 0.001 (measured in K2)"
    else:
        decision["passed_k1_gates"] = False
    substitutions = {"address": Counter(), "name": Counter()}
    for path in sorted((work / "sample").glob("chunk*.subst.json")):
        for field, rows in json.loads(path.read_text()).items():
            for a, b, count in rows:
                substitutions[field][a, b] += count
    atomic_write_json(output / "substitutions_train_gold.json",
                      {field: [[a, b, n] for (a, b), n in counter.most_common(5000)] for field, counter in substitutions.items()})
    ownership_table = defaultdict(dict)
    for key, value in ownership.items():
        stratum, kind = key.split("|")
        ownership_table[stratum][kind] = value
    for stratum, row in ownership_table.items():
        negatives = row.get("neg_unowned", 0) + row.get("neg_owned_by_other_s1", 0)
        row["share_of_negatives_owned_by_other_s1"] = row.get("neg_owned_by_other_s1", 0) / negatives if negatives else None
        row["positive_rate"] = row.get("positive", 0) / (negatives + row.get("positive", 0)) if negatives + row.get("positive", 0) else None
    gold = patterns.get("gold_pairs", 0)
    training = json.loads((output / "models" / "training.json").read_text())
    result = {
        "run_id": config["run_id"], "decision": decision, "by_split": by_split, "transfer_us_to_india_validation": transfer_curve,
        "filter_training": training,
        "analysis": {
            "ownership_validation_holdout": dict(ownership_table),
            "gold_patterns_validation_holdout": {k: {"count": v, "share": v / gold if gold else None}
                                                 for k, v in patterns.most_common() if k != "gold_pairs"},
            "gold_pairs": gold, "gold_target_name_script": dict(scripts.most_common()),
            "gold_name_token_set_bins": dict(name_bins),
            "address_substitutions_top": [[a, b, n] for (a, b), n in substitutions["address"].most_common(40)],
            "name_substitutions_top": [[a, b, n] for (a, b), n in substitutions["name"].most_common(25)],
            "dropped_gold_examples_rank_ge_60": examples[:40],
        },
        "score_stage_seconds": seconds,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    }
    atomic_write_json(output / "k1_results.json", result)
    (output / "K1_REPORT.md").write_text(make_report(result, config))
    return result


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.3f}%"


def make_report(result: dict, config: dict) -> str:
    d = result["decision"]
    lines = ["# K1 — learned first-stage filter (experiment) and full-fold error structure", "",
             "Frozen B+C+E blocker candidates for all 441,467 fold-0 S1 (Kaggle Phase 1C shards, content-hash verified). "
             "Split by S1: train 0–330,999 (out-of-fold filter scores), validation 331,000–385,999, holdout 386,000–441,466.", "",
             f"**Filter decision:** chosen K = {d.get('chosen_k')}; K1 gates passed = **{d.get('passed_k1_gates')}** "
             f"(retention target {pct(d['retention_target'])}). Remaining gate: {d.get('pending_gate', 'n/a')}.", ""]
    if d.get("chosen_k") is not None:
        lines += ["| Measure at chosen K | Value |", "|---|---:|",
                  f"| Validation gold retention | {pct(d['validation_retention'])} |",
                  f"| Holdout gold retention | {pct(d['holdout_retention'])} |",
                  f"| Transfer (US-trained filter on India validation) | {pct(d['transfer_retention'])} |",
                  f"| Oracle macro F0.5 at K vs no filter (validation) | {d['validation_oracle_f05_at_k']:.4f} vs {d['validation_oracle_f05_no_filter']:.4f} |",
                  f"| Mean kept candidates per S1 | {d['mean_kept_candidates']:.1f} |",
                  f"| Projected test pairs (with / without filter) | {d['projected_test_pairs']/1e6:.0f}M / {d['projected_test_pairs_no_filter']/1e6:.0f}M |", ""]
    lines += ["## Retention curve (share of blocker-retrieved gold kept in the top K)", "",
              "| K | " + " | ".join(f"{s} retention" for s in ("train (OOF)", "validation", "holdout")) + " | US→India transfer | Val oracle F0.5 | Mean kept |",
              "|---:|" + "---:|" * 6]
    for k in config["k_grid"]:
        row = [by["all"]["curve"][str(k)]["retention"] for by in (result["by_split"][s] for s in SPLITS)]
        transfer = result["transfer_us_to_india_validation"][str(k)]["retention"] if result["transfer_us_to_india_validation"] else None
        v = result["by_split"]["validation"]["all"]["curve"][str(k)]
        lines.append(f"| {k} | " + " | ".join(pct(x) for x in row) + f" | {pct(transfer)} | {v['oracle_macro_f05']:.4f} | {v['mean_kept_candidates']:.1f} |")
    lines += ["", "## Hard negatives: who owns them? (validation + holdout pairs)", "",
              "Each S2/S3 record belongs to at most one S1. 'Owned by other S1' negatives are resolvable by one-S1-per-record logic when that S1 is also scored.", "",
              "| Stratum | Positives | Negatives unowned | Negatives owned by another S1 | Share owned | Positive rate |", "|---|---:|---:|---:|---:|---:|"]
    for stratum, row in result["analysis"]["ownership_validation_holdout"].items():
        lines.append(f"| {stratum} | {row.get('positive', 0):,} | {row.get('neg_unowned', 0):,} | {row.get('neg_owned_by_other_s1', 0):,} | "
                     f"{pct(row['share_of_negatives_owned_by_other_s1'])} | {pct(row['positive_rate'])} |")
    a = result["analysis"]
    lines += ["", f"## Gold-pair noise patterns (validation + holdout, {a['gold_pairs']:,} gold pairs)", "",
              "| Pattern | Share |", "|---|---:|"]
    lines += [f"| {k} | {pct(v['share'])} |" for k, v in a["gold_patterns_validation_holdout"].items()]
    lines += ["", "Target name scripts: " + ", ".join(f"{k} {v:,}" for k, v in a["gold_target_name_script"].items()) + ".",
              "Name token-set similarity bins: " + ", ".join(f"{k} {v:,}" for k, v in a["gold_name_token_set_bins"].items()) + ".",
              "", "Top address word substitutions mined from train gold pairs: " +
              ", ".join(f"{x}→{y} ({n:,})" for x, y, n in a["address_substitutions_top"][:25]) + ".", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1c-work-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("stores", "sample", "train", "score", "report", "all"), required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.resolve().read_text())
    resources = {"min_available_memory_gib": 1.0, "max_swap_growth_gib": 4.0}
    command = (f"python3 -m src.k1_filter --config {args.config} --phase1c-work-dir {args.phase1c_work_dir} "
               f"--work-dir {args.work_dir} --output-dir {args.output_dir} --stage {args.stage}")
    guard = ResourceGuard(resources["min_available_memory_gib"], resources["max_swap_growth_gib"], command)
    started = time.perf_counter()
    try:
        ctx = load_context(config, ROOT, args.phase1c_work_dir)
        ctx["train_dir"] = ROOT / config["inputs"]["train_dir"]
        log(f"context: {len(ctx['ordered']):,} fold S1; countries {ctx['keys']}")
        stages = ("stores", "sample", "train", "score", "report") if args.stage == "all" else (args.stage,)
        stores = shards = None
        for stage in stages:
            if stage == "stores":
                build_stores(config, ROOT, args.work_dir, ctx, guard)
                continue
            if stage in ("sample", "score") and stores is None:
                stores = Stores(args.work_dir)
                manifest = ctx["manifest"]
                shards = Phase1CShards(args.phase1c_work_dir, manifest)
            if stage == "sample":
                stage_sample(config, ctx, stores, shards, args.work_dir, guard)
            elif stage == "train":
                stage_train(config, ctx, args.work_dir, args.output_dir, guard)
            elif stage == "score":
                stage_score(config, ctx, stores, shards, args.work_dir, args.output_dir, guard)
            elif stage == "report":
                result = stage_report(config, ctx, args.work_dir, args.output_dir)
                log("report: " + json.dumps(result["decision"]))
    except ResourceStop as stop:
        log(str(stop))
        raise SystemExit(3)
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "stage_runs.jsonl").open("a") as file:
            file.write(json.dumps({"stage": args.stage, "seconds": time.perf_counter() - started, **guard.summary()}) + "\n")


if __name__ == "__main__":
    main()
