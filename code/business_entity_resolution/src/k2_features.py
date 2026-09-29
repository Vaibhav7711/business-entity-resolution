"""K2 stage-2 features on filtered candidate lists for every fold-0 S1 (multi-process).

For each fold-0 S1 the frozen blocker's candidate union (Phase 1C shards) is re-scored by
the K1 first-stage filter (out-of-fold model for train S1, final model otherwise) and cut
to the top K. Every kept pair then gets four feature groups:

* ``v1``: the 42 Phase 2A features (retrieval evidence and fuzzy string similarity);
* ``norm``: hashed overlaps from K1 (numbers with leading zeros removed, IDF-weighted
  name/address words) plus v2 features (one-digit typos, dropped number components,
  word substitutions mined from *training* gold pairs, Latin-only accent folding, and
  name scripts);
* ``context``: the pair's standing inside its own S1 list (filter score rank and gap,
  number of strong candidates, co-located candidates, and similarity to the top
  candidate) and the corpus frequency of its address number key.

Nothing is computed across S1 records, so every feature is identical at train and test time.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import json
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from .evaluate_phase1c import atomic_savez, atomic_write_json
from .k1_filter import (
    CHEAP_FEATURES, DIGITS, Phase1CShards, Stores, build_chunk, cheap_features, overlap, rank_within_s1,
    ratio_or_nan, shard_keys,
)
from .phase2a_env import log
from .phase2a_pairs import FEATURES as V1_FEATURES, compute_features, split_of

SCRIPTS = ("LATIN", "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA",
           "MALAYALAM", "ARABIC", "OTHER")
NORM_EXTRA = tuple(name for name in CHEAP_FEATURES if name not in V1_FEATURES)
V2_FEATURES = ("digits_one_edit", "tgt_digits_all_in_s1", "name_canon_token_set", "addr_canon_token_set",
               "addr_canon_ratio", "s1_name_script", "tgt_name_script", "name_script_mismatch")
CONTEXT_FEATURES = ("filter_score", "filter_rank", "filter_rank_frac", "filter_gap_to_top", "filter_score_frac",
                    "list_n_strong", "list_size", "list_n_s3", "same_address_in_list", "name_rank_same_address",
                    "best_other_name_idf_same_address", "top1_name_overlap", "top1_digits_overlap",
                    "tgt_digit_key_log_freq")
ALL_FEATURES = V1_FEATURES + NORM_EXTRA + V2_FEATURES + CONTEXT_FEATURES
GROUPS = {"v1": V1_FEATURES, "norm": NORM_EXTRA + V2_FEATURES, "context": CONTEXT_FEATURES}
CHEAP_INDEX = {name: i for i, name in enumerate(CHEAP_FEATURES)}


# ---------------------------------------------------------------------------
# Text normalisation helpers


def fold_latin(text: str) -> str:
    """Remove accents from Latin letters only (keeps Indic vowel signs and other scripts intact)."""
    out = []
    for char in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(char) and out and out[-1].isascii():
            continue
        out.append(char)
    return unicodedata.normalize("NFKC", "".join(out))


def script_code(text: str) -> int:
    for char in text:
        if char.isalpha() and ord(char) > 127:
            name = unicodedata.name(char, "")
            script = name.split(" ")[0] if name else "OTHER"
            return SCRIPTS.index(script) if script in SCRIPTS else SCRIPTS.index("OTHER")
        if char.isalpha():
            return 0
    return 0


def build_canon_map(substitutions: dict, min_count: int, min_share: float) -> dict[str, dict[str, str]]:
    """Token -> canonical token, from substitutions mined on training gold pairs.

    A token is merged with its dominant partner when that partner accounts for at least
    ``min_share`` of its substitutions and occurs ``min_count`` times. Merged groups are
    joined with union-find, and the lexicographically smallest member names each group.
    """
    maps = {}
    for field, rows in substitutions.items():
        total, best = Counter(), {}
        for a, b, count in rows:
            total[a] += count
            if count > best.get(a, ("", 0))[1]:
                best[a] = (b, count)
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x

        for a, (b, count) in best.items():
            if count >= min_count and count / total[a] >= min_share:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        maps[field] = {token: find(token) for token in set(parent) | {p for p in parent.values()} if find(token) != token}
    return maps


def canonical(text: str, mapping: dict[str, str]) -> str:
    return " ".join(mapping.get(token, token) for token in fold_latin(text).split())


def canon_digits(text: str) -> list[str]:
    return list(dict.fromkeys(token.lstrip("0") or "0" for token in DIGITS.findall(text)))


def digit_key(digits: np.ndarray) -> np.ndarray:
    """Order-free uint64 key of a record's number-token hashes (0 when it has none)."""
    d = np.sort(np.asarray(digits, dtype=np.uint64), axis=1)
    key = np.zeros(len(d), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(d.shape[1]):
            key = key * np.uint64(1_000_003) + d[:, j]
    key[(d == 0).all(axis=1)] = 0
    return key


# ---------------------------------------------------------------------------
# Feature groups


class TargetAdapter:
    """Presents k1 ``Stores`` through the interface Phase 2A's ``compute_features`` expects."""

    def __init__(self, stores: Stores):
        self.stores = stores
        self.meta = {"address_missing": stores.t["addr_missing"], "name_missing": stores.t["name_missing"],
                     "non_ascii": stores.t["non_ascii"]}

    def rows(self, cand: np.ndarray) -> np.ndarray:
        return self.stores.rows(cand)

    def text(self, row: int, field: str) -> str:
        return self.stores.text(row, "addr" if field == "address" else "name")


def v2_features(kept: dict, ctx: dict, stores: Stores, canon: dict, cache: dict) -> np.ndarray:
    q = ctx["queries"]
    cand, positions = kept["cand"], kept["s1_pos"].astype(np.int64)
    trows = stores.rows(cand)

    def s1_info(p: int):
        info = cache["s1"].get(p)
        if info is None:
            info = cache["s1"][p] = (canon_digits(q["address"][p]), canonical(q["name"][p], canon.get("name", {})),
                                     canonical(q["address"][p], canon.get("address", {})), script_code(q["name"][p]))
        return info

    def t_info(row: int):
        info = cache["t"].get(row)
        if info is None:
            name, address = stores.text(row, "name"), stores.text(row, "addr")
            info = cache["t"][row] = (canon_digits(address), canonical(name, canon.get("name", {})),
                                      canonical(address, canon.get("address", {})), script_code(name))
        return info

    n = len(cand)
    one_edit = np.zeros(n, np.float32)
    subset = np.full(n, np.nan, np.float32)
    s_script = np.zeros(n, np.float32)
    t_script = np.zeros(n, np.float32)
    q_name, q_addr, t_name, t_addr = [], [], [], []
    for i in range(n):
        a = s1_info(int(positions[i]))
        b = t_info(int(trows[i]))
        da, db = set(a[0]), set(b[0])
        only_a, only_b = da - db, db - da
        one_edit[i] = any(len(x) == len(y) and Levenshtein.distance(x, y) == 1 for x in only_a for y in only_b)
        if db:
            subset[i] = float(db <= da)
        q_name.append(a[1]); q_addr.append(a[2]); t_name.append(b[1]); t_addr.append(b[2])
        s_script[i], t_script[i] = a[3], b[3]
    workers = phase2a_env.WORKERS
    both_name = np.asarray([bool(x) and bool(y) for x, y in zip(q_name, t_name)])
    both_addr = np.asarray([bool(x) and bool(y) for x, y in zip(q_addr, t_addr)])
    name_ts = np.asarray(process.cpdist(q_name, t_name, scorer=fuzz.token_set_ratio, workers=workers), np.float32) / 100
    addr_ts = np.asarray(process.cpdist(q_addr, t_addr, scorer=fuzz.token_set_ratio, workers=workers), np.float32) / 100
    addr_r = np.asarray(process.cpdist(q_addr, t_addr, scorer=fuzz.ratio, workers=workers), np.float32) / 100
    return np.column_stack([
        one_edit, subset, np.where(both_name, name_ts, np.nan), np.where(both_addr, addr_ts, np.nan),
        np.where(both_addr, addr_r, np.nan), s_script, t_script, (s_script != t_script).astype(np.float32),
    ]).astype(np.float32)


def group_bounds(s1_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows are contiguous per S1: return group index per row, group starts, and sizes."""
    starts = np.flatnonzero(np.r_[True, s1_pos[1:] != s1_pos[:-1]])
    sizes = np.diff(np.r_[starts, len(s1_pos)])
    return np.repeat(np.arange(len(starts)), sizes), starts, sizes


def context_features(kept: dict, fscore: np.ndarray, frank: np.ndarray, cheap: np.ndarray, stores: Stores,
                     density: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    g, starts, sizes = group_bounds(kept["s1_pos"])
    n = len(fscore)
    top = np.maximum.reduceat(fscore, starts) if n else np.zeros(0)
    size = sizes[g].astype(np.float32)
    n_strong = np.add.reduceat((fscore >= 0.5).astype(np.int64), starts)[g] if n else np.zeros(0)
    n_s3 = np.add.reduceat((kept["cand"] & 1).astype(np.int64), starts)[g] if n else np.zeros(0)
    trows = stores.rows(kept["cand"])
    tdig = np.asarray(stores.t["digits"][trows])
    key = digit_key(tdig)
    # Co-located candidates within the same S1 list (same number key).
    order = np.lexsort((-np.nan_to_num(cheap[:, CHEAP_INDEX["name_idf_frac"]]), key, g))
    gk = np.stack([g[order], key[order].view(np.int64)], axis=1)
    new = np.r_[True, (gk[1:] != gk[:-1]).any(axis=1)] if n else np.zeros(0, bool)
    block_start = np.flatnonzero(new)
    block_size = np.diff(np.r_[block_start, n])
    block_of = np.repeat(np.arange(len(block_start)), block_size)
    same = np.empty(n, np.float32)
    rank_same = np.empty(n, np.float32)
    best_other = np.empty(n, np.float32)
    name_idf_sorted = np.nan_to_num(cheap[order, CHEAP_INDEX["name_idf_frac"]])
    within = np.arange(n) - block_start[block_of]
    first = name_idf_sorted[block_start][block_of]
    second = np.where(block_size[block_of] > 1, name_idf_sorted[np.minimum(block_start + 1, n - 1)][block_of], np.nan)
    same[order] = block_size[block_of] - 1
    rank_same[order] = within
    best_other[order] = np.where(within == 0, second, first)
    no_key = key == 0
    same[no_key], rank_same[no_key], best_other[no_key] = np.nan, np.nan, np.nan
    # Similarity to the S1's top candidate (by filter score; ties by candidate ID).
    top_order = np.lexsort((kept["cand"], -fscore, g))
    top_row = np.empty(len(starts), dtype=np.int64)
    top_row[g[top_order][np.r_[True, g[top_order][1:] != g[top_order][:-1]]]] = top_order[np.r_[True, g[top_order][1:] != g[top_order][:-1]]]
    top_trow = trows[top_row][g]
    this_names = np.asarray(stores.t["name"][trows])
    top_names = np.asarray(stores.t["name"][top_trow])
    name_ov, _ = overlap(this_names, top_names)
    digit_ov, _ = overlap(tdig, np.asarray(stores.t["digits"][top_trow]))
    uniq, counts = density
    where = np.minimum(np.searchsorted(uniq, key), max(len(uniq) - 1, 0))
    freq = np.where((key != 0) & (uniq[where] == key), counts[where], 0) if len(uniq) else np.zeros(n)
    return np.column_stack([
        fscore, frank, frank / np.maximum(size, 1), top[g] - fscore, ratio_or_nan(fscore, top[g]),
        n_strong, size, n_s3, same, rank_same, best_other,
        ratio_or_nan(name_ov, (this_names != 0).sum(axis=1)), ratio_or_nan(digit_ov, (tdig != 0).sum(axis=1)),
        np.where(key != 0, np.log1p(freq), np.nan),
    ]).astype(np.float32)


def density_table(stores: Stores, work: Path) -> tuple[np.ndarray, np.ndarray]:
    """Corpus frequency of every target number key (computed once from the searched corpus)."""
    path = work / "stores" / "digit_key_density.npz"
    if not path.exists():
        digits = stores.t["digits"]
        keys = np.concatenate([digit_key(np.asarray(digits[i:i + 2_000_000])) for i in range(0, len(digits), 2_000_000)])
        uniq, counts = np.unique(keys[keys != 0], return_counts=True)
        atomic_savez(path, {"uniq": uniq, "counts": counts.astype(np.int64)})
    with np.load(path) as data:
        return data["uniq"], data["counts"]


# ---------------------------------------------------------------------------
# Chunk driver (one process per Phase 1C shard)


_STATE: dict = {}


def init_worker(config: dict, ctx: dict, phase1c_work: str, work: str, k1_dir: str, canon: dict, keep_k: int | None,
                s1_dir: str | None = None, features_dir: str = "features", filter_all_final: bool = False,
                export: dict | None = None) -> None:
    import lightgbm as lgb

    stores = Stores(Path(work), Path(s1_dir) if s1_dir else None)
    _STATE.update(config=config, ctx=ctx, stores=stores, adapter=TargetAdapter(stores), canon=canon, keep_k=keep_k,
                  features_dir=features_dir, filter_all_final=filter_all_final, export=export,
                  shards=Phase1CShards(Path(phase1c_work), ctx["manifest"]), work=Path(work),
                  density=density_table(stores, Path(work)),
                  models={name: lgb.Booster(model_file=str(Path(k1_dir) / "models" / f"filter_{name}.txt"))
                          for name in [f"fold{f}" for f in range(config["oof_folds"])] + ["final"]})
    q = ctx["queries"]
    _STATE["s1"] = {"name": q["name"], "address": q["address"],
                    "address_missing": np.asarray([not a for a in q["address"]]), "non_ascii": q["non_ascii"]}


def process_shard(shard: int) -> dict:
    s = _STATE
    config, ctx, stores = s["config"], s["ctx"], s["stores"]
    size = ctx["p1c"]["algorithm"]["shard_size"]
    chunk = config["chunk_s1"]
    n = len(ctx["ordered"])
    low, high = shard * size, min((shard + 1) * size, n)
    out_dir = s["work"] / s["features_dir"]
    todo = [start for start in range(low, high, chunk) if not (out_dir / f"chunk{start // chunk:03d}.npz").exists()]
    if not todo:
        return {"shard": shard, "chunks": 0}
    loaded, locator = s["shards"].load(shard, shard_keys(ctx, shard, size))
    folds = config["oof_folds"]
    train_end = config["split"]["train"][1]
    cache = {"s1": {}, "t": {}}
    started = time.perf_counter()
    for start in todo:
        stop = min(start + chunk, high)
        arrays = build_chunk(start, stop, loaded, locator, ctx, config["rrf_constant"])
        rows = np.arange(len(arrays["cand"]))
        cheap = cheap_features(arrays, rows, stores, start)
        pos = arrays["s1_pos"].astype(np.int64)
        score = np.empty(len(pos))
        is_train = (pos < train_end) & (not s["filter_all_final"])
        for f in range(folds):
            mask = is_train & (pos % folds == f)
            if mask.any():
                score[mask] = s["models"][f"fold{f}"].predict(cheap[mask], num_threads=1)
        if (~is_train).any():
            score[~is_train] = s["models"]["final"].predict(cheap[~is_train], num_threads=1)
        rank = rank_within_s1(pos, score, arrays["cand"].astype(np.int64))
        keep = rank < s["keep_k"] if s["keep_k"] else np.ones(len(rank), bool)
        if s.get("export") and s["export"].get("topk_only"):
            atomic_savez(out_dir / f"chunk{start // chunk:03d}.npz", {
                "s1_pos": arrays["s1_pos"][keep].astype(np.int64), "cand": arrays["cand"][keep],
                "label": arrays["label"][keep], "filter_score": score[keep].astype(np.float32),
                "filter_rank": rank[keep].astype(np.int16), "s1_positions": arrays["s1_positions"],
                "s1_truth_len": arrays["s1_truth_len"], "s1_retrieved_truth": arrays["s1_retrieved_truth"]})
            continue
        kept = {name: arrays[name][keep] for name in ("s1_pos", "cand", "label", "route_bits", "rrf", "rrf_rank",
                                                      "cand_count", *[k for k in arrays if k.startswith(("score_", "rank_"))])}
        kept["s1_positions"], kept["s1_exact_hits"] = arrays["s1_positions"], arrays["s1_exact_hits"]
        kept_rows = np.arange(len(kept["cand"]))
        v1 = compute_features(kept, kept_rows, s["s1"], s["adapter"])
        extra = cheap[keep][:, [CHEAP_INDEX[name] for name in NORM_EXTRA]]
        v2 = v2_features(kept, ctx, stores, s["canon"], cache)
        context = context_features(kept, score[keep].astype(np.float32), rank[keep].astype(np.float32), cheap[keep],
                                   stores, s["density"])
        X = np.column_stack([v1, extra, v2, context]).astype(np.float32)
        if X.shape[1] != len(ALL_FEATURES) or np.isinf(X).any():
            raise AssertionError("stage-2 feature matrix shape/infinity check failed")
        kept_truth = np.bincount(kept["s1_pos"][kept["label"]] - start, minlength=stop - start)
        payload = {
            "X": X, "label": kept["label"], "s1_pos": kept["s1_pos"].astype(np.int64), "cand": kept["cand"],
            "s1_positions": arrays["s1_positions"], "s1_truth_len": arrays["s1_truth_len"],
            "s1_retrieved_truth": arrays["s1_retrieved_truth"], "s1_kept_truth": kept_truth,
            "s1_cand_count": arrays["s1_cand_count"]}
        if s.get("export"):
            atomic_savez_compressed(out_dir / f"chunk{start // chunk:03d}.npz",
                                    export_rows(payload, s["export"], start // chunk, ctx))
        else:
            atomic_savez(out_dir / f"chunk{start // chunk:03d}.npz", payload)
        if len(cache["t"]) > 400_000:
            cache["t"].clear()
    return {"shard": shard, "chunks": len(todo), "seconds": time.perf_counter() - started}


def export_rows(data: dict, export: dict, index: int, ctx: dict) -> dict:
    """R4 export of one chunk: K2's sampled training rows with their natural-rate weights (exactly what
    ``k2_experiments.load_split(..., sample=True)`` would draw), or every row for validation/holdout chunks.
    Adds per-row country codes and per-S1 country codes and integer IDs (``s1_key``)."""
    from .k2_experiments import sample_rows

    first = int(data["s1_positions"][0]) if len(data["s1_positions"]) else 0
    if export["train_all"] or first < export["train_end"]:
        rows, weight = sample_rows(data, export["seed"] + export["offset"], index, export["top"], export["random"])
    else:
        rows, weight = np.arange(len(data["label"])), np.ones(len(data["label"]), np.float32)
    local = data["s1_pos"][rows]
    positions = data["s1_positions"]
    return {"X": data["X"][rows], "label": data["label"][rows], "s1_pos": local, "cand": data["cand"][rows],
            "weight": weight, "country": ctx["country_code"][local],
            "s1_positions": positions, "s1_truth_len": data["s1_truth_len"], "s1_retrieved_truth": data["s1_retrieved_truth"],
            "s1_kept_truth": data["s1_kept_truth"], "s1_cand_count": data["s1_cand_count"],
            "s1_country": ctx["country_code"][positions], "s1_key": ctx["s1_num"][positions]}


def atomic_savez_compressed(path: Path, arrays: dict) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary, path)


def build_features(config: dict, ctx: dict, phase1c_work: Path, work: Path, k1_dir: Path, keep_k: int | None,
                   workers: int, *, s1_dir: Path | None = None, features_dir: str = "features",
                   filter_all_final: bool = False, export: dict | None = None) -> None:
    """Run ``process_shard`` over all Phase 1C shards with ``workers`` processes (fork).

    Extra training folds pass their own ``s1_dir``/``features_dir`` and ``filter_all_final=True``: their S1
    never trained the K1 filter, so the final filter model scores them like fold-0 out-of-fold rows.
    ``export`` (R4) writes compressed chunks holding only what R5 needs (see ``export_rows``); an absolute
    ``features_dir`` puts them outside the work directory.
    """
    import multiprocessing as mp

    (work / features_dir).mkdir(parents=True, exist_ok=True)
    substitutions = json.loads((k1_dir / "substitutions_train_gold.json").read_text())
    canon = build_canon_map(substitutions, config["substitutions"]["min_count"], config["substitutions"]["min_share"])
    atomic_write_json(work / "canon_map.json", canon)
    stores = Stores(work, s1_dir)
    density_table(stores, work)
    size = ctx["p1c"]["algorithm"]["shard_size"]
    n_shards = (len(ctx["ordered"]) + size - 1) // size
    args = (config, ctx, str(phase1c_work), str(work), str(k1_dir), canon, keep_k, str(s1_dir) if s1_dir else None,
            features_dir, filter_all_final, export)
    started = time.perf_counter()
    if workers <= 1:
        init_worker(*args)
        for shard in range(n_shards):
            info = process_shard(shard)
            log(f"features: shard {info['shard']} ({info['chunks']} chunks)")
    else:
        context = mp.get_context("fork")
        with context.Pool(workers, initializer=init_worker, initargs=args) as pool:
            for info in pool.imap_unordered(process_shard, range(n_shards)):
                log(f"features: shard {info['shard']} ({info['chunks']} chunks, {info.get('seconds', 0):.0f}s)")
    log(f"features: all shards done in {(time.perf_counter() - started) / 60:.1f} min")
