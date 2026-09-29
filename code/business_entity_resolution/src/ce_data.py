"""Export (S1, candidate) text pairs for the cross-encoder, keyed so scores join back to K2 rows exactly.

Sets (fold-0 validation/holdout are the untouched yardstick):

* ``ce_train``: sampled pairs from ``ce.train_folds`` (default folds 1-2). The cross-encoder learns only
  from these S1.
* ``stack_train``: exactly K2's sampled training rows for fold-0 train plus ``ce.stack_folds`` (default
  3-4). The cross-encoder never saw these S1, so its scores are honest stacking features.
* ``validation`` / ``holdout``: every kept pair of those fold-0 ranges.
* ``test``: written by K3 (``export`` stage) in the same format.

Each row is (fold, s1_pos [fold-local], cand, label, text_a, text_b). ``text_a`` is the S1 name | address
and ``text_b`` is the candidate name | address, both with the pipeline's normalisation (scripts preserved).
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401

import json
from pathlib import Path

import numpy as np

from .k1_filter import Stores
from .k2_experiments import sample_rows
from .phase2a_env import log

PARQUET_ROWS = 2_000_000


def pair_text(name: str, address: str) -> str:
    return f"{name} | {address}" if address else name


def write_parquet(path: Path, columns: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "fold": pa.array(columns["fold"], pa.int8()), "s1_pos": pa.array(columns["s1_pos"], pa.int64()),
        "cand": pa.array(columns["cand"], pa.uint32()), "label": pa.array(columns["label"], pa.bool_()),
        "text_a": pa.array(columns["text_a"], pa.string()), "text_b": pa.array(columns["text_b"], pa.string())})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


class PairWriter:
    """Buffers rows and writes numbered parquet parts of about ``PARQUET_ROWS`` rows."""

    def __init__(self, directory: Path):
        self.directory, self.part, self.rows = directory, 0, 0
        self.buffer = {k: [] for k in ("fold", "s1_pos", "cand", "label", "text_a", "text_b")}

    def add(self, fold: int, s1_pos: np.ndarray, cand: np.ndarray, label: np.ndarray, text_a: list, text_b: list) -> None:
        self.buffer["fold"].extend([fold] * len(cand)); self.buffer["s1_pos"].extend(s1_pos.tolist())
        self.buffer["cand"].extend(cand.tolist()); self.buffer["label"].extend(label.tolist())
        self.buffer["text_a"].extend(text_a); self.buffer["text_b"].extend(text_b)
        if len(self.buffer["cand"]) >= PARQUET_ROWS:
            self.flush()

    def flush(self) -> None:
        if self.buffer["cand"]:
            write_parquet(self.directory / f"part-{self.part:05d}.parquet", self.buffer)
            self.rows += len(self.buffer["cand"])
            self.part += 1
            self.buffer = {k: [] for k in self.buffer}

    def close(self) -> dict:
        self.flush()
        return {"parts": self.part, "rows": self.rows}


def export_rows(writer: PairWriter, fold: int, data: dict, rows: np.ndarray, queries: dict, stores: Stores,
                cache: dict) -> None:
    s1_pos, cand = data["s1_pos"][rows], data["cand"][rows]
    text_a = [pair_text(queries["name"][int(p)], queries["address"][int(p)]) for p in s1_pos]
    trows = stores.rows(cand)
    text_b = []
    for row in trows.tolist():
        text = cache.get(row)
        if text is None:
            text = cache[row] = pair_text(stores.text(row, "name"), stores.text(row, "addr"))
        text_b.append(text)
    writer.add(fold, s1_pos, cand, data["label"][rows], text_a, text_b)
    if len(cache) > 2_000_000:
        cache.clear()


def export_set(name: str, out_dir: Path, sources: list[tuple], config: dict, stores: Stores, *, sample: bool,
               max_s1: int | None = None) -> dict:
    """``sources``: (fold, features_dir, chunk_indices, queries, offset) for each fold feeding this set."""
    done = out_dir / name / "DONE.json"
    if done.exists():
        return json.loads(done.read_text())
    writer = PairWriter(out_dir / name)
    s = config["train_sampling"]
    cache: dict = {}
    s1_seen = 0
    for fold, features_dir, indices, queries, offset in sources:
        for index in indices:
            if max_s1 is not None and s1_seen >= max_s1:
                break
            with np.load(features_dir / f"chunk{index:03d}.npz", allow_pickle=False) as data:
                data = {k: data[k] for k in ("s1_pos", "cand", "label", "X", "s1_positions")}
            if sample:
                rows, _ = sample_rows(data, s["seed"] + offset, index, s["top_filter_negatives"], s["random_negatives"])
            else:
                rows = np.arange(len(data["label"]))
            export_rows(writer, fold, data, rows, queries, stores, cache)
            s1_seen += len(data["s1_positions"])
        log(f"ce_data: {name} fold {fold} exported")
    info = writer.close() | {"s1": s1_seen}
    done.write_text(json.dumps(info))
    return info


def export_all(config: dict, ctx: dict, extra: list, work: Path, out_dir: Path) -> dict:
    """Write ce_train, stack_train, validation, and holdout pair sets under ``out_dir``."""
    ce = config["ce"]
    chunk = config["chunk_s1"]
    by_fold = {e["fold"]: e for e in extra}
    summary = {}
    stores = Stores(work)
    train_sources = []
    for fold in ce["train_folds"]:
        e = by_fold[fold]
        n_k = len(e["ctx"]["ordered"])
        train_sources.append((fold, work / e["features_dir"], range((n_k + chunk - 1) // chunk), e["ctx"]["queries"], e["offset"]))
    summary["ce_train"] = export_set("ce_train", out_dir, train_sources, config, stores, sample=True,
                                     max_s1=ce.get("max_train_s1"))
    low, high = config["split"]["train"]
    stack_sources = [(0, work / "features", range(low // chunk, high // chunk), ctx["queries"], 0)]
    for fold in ce["stack_folds"]:
        e = by_fold[fold]
        n_k = len(e["ctx"]["ordered"])
        stack_sources.append((fold, work / e["features_dir"], range((n_k + chunk - 1) // chunk), e["ctx"]["queries"], e["offset"]))
    summary["stack_train"] = export_set("stack_train", out_dir, stack_sources, config, stores, sample=True)
    for split in ("validation", "holdout"):
        low, high = config["split"][split]
        summary[split] = export_set(split, out_dir, [(0, work / "features", range(low // chunk, (high + chunk - 1) // chunk),
                                                      ctx["queries"], 0)], config, stores, sample=False)
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_scores(score_dir: Path) -> dict:
    """Concatenate cross-encoder score parts: fold, s1_pos (fold-local), cand, logit."""
    parts = {k: [] for k in ("fold", "s1_pos", "cand", "logit")}
    for path in sorted(score_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for k in parts:
                parts[k].append(data[k])
    if not parts["cand"]:
        raise FileNotFoundError(f"no cross-encoder scores under {score_dir}")
    return {k: np.concatenate(v) for k, v in parts.items()}


def join_scores(scores: dict, fold_offsets: dict[int, int], s1_pos: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Cross-encoder logit for each (offset position, candidate) row; raises if any row is missing."""
    offsets = np.asarray([fold_offsets[int(f)] for f in scores["fold"]], dtype=np.int64) if len(scores["fold"]) else np.zeros(0, np.int64)
    key_scores = ((scores["s1_pos"].astype(np.int64) + offsets) << 32) | scores["cand"].astype(np.int64)
    order = np.argsort(key_scores, kind="stable")
    sorted_keys, sorted_logits = key_scores[order], scores["logit"][order]
    keys = (s1_pos.astype(np.int64) << 32) | cand.astype(np.int64)
    where = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    if not np.array_equal(sorted_keys[where], keys):
        missing = int((sorted_keys[where] != keys).sum())
        raise KeyError(f"{missing} rows have no cross-encoder score")
    return sorted_logits[where].astype(np.float32)
