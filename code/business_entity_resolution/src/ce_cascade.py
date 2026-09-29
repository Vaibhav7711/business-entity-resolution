"""Cascade pairs for the round-2 cross-encoder: each S1's top-N candidates by the round-1 cross-encoder.

Reads round 1's pairs root + ``<ce1_out>/scores/<split>.npy`` (NaN = not scored; such rows are dropped) and writes a
new pairs root in the same format (see ``topk_export``), where ``filter_score`` is the round-1 logit and
``filter_rank`` its rank within the S1 (0 = best), so the policy's stacked variant combines both models.

N is chosen on validation: the smallest N whose top-N keeps ``retention`` of the true matches that round 1 had in its
lists (recall already lost upstream stays lost; ``truth_len`` is copied unchanged so the metric still counts it).
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_phase1c import atomic_write_json
from .phase2a_env import log

PART_ROWS = 4_000_000


def read_split(pairs_root: Path, scores_dir: Path, split: str):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = pairs_root / split
    parts = sorted(directory.glob("part-*.parquet"))
    table = pa.concat_tables([pq.read_table(p, columns=["s1_id", "t_id", "label"]) for p in parts])
    logit = np.load(scores_dir / f"{split}.npy")
    if len(logit) != table.num_rows:
        raise ValueError(f"{split}: {len(logit):,} scores for {table.num_rows:,} rows")
    return table, logit, pq.read_table(directory / "s1.parquet")


def cascade_rank(s1_codes: np.ndarray, logit: np.ndarray) -> np.ndarray:
    """Rank of each row's logit within its S1 (0 = best; NaN rows get a large rank)."""
    key = np.where(np.isfinite(logit), -logit, np.inf)
    order = np.lexsort((key, s1_codes))
    rank = np.empty(len(order), np.int64)
    starts = np.r_[0, np.flatnonzero(s1_codes[order][1:] != s1_codes[order][:-1]) + 1]
    run = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(order)]))
    rank[order] = np.arange(len(order)) - starts[run]
    rank[~np.isfinite(logit)] = 1 << 30
    return rank


def choose_n(pairs_root: Path, scores_dir: Path, retention: float, max_n: int) -> dict:
    table, logit, _ = read_split(pairs_root, scores_dir, "validation")
    codes = table.column("s1_id").combine_chunks().dictionary_encode().indices.to_numpy()
    label = table.column("label").to_numpy().astype(bool)
    rank = cascade_rank(codes, logit)
    positives = int(label[np.isfinite(logit)].sum())
    curve = {n: float(label[rank < n].sum() / max(positives, 1)) for n in range(1, max_n + 1)}
    chosen = next((n for n in range(1, max_n + 1) if curve[n] >= retention), max_n)
    return {"n": chosen, "retention_target": retention, "curve": curve, "validation_positives_in_lists": positives}


def write_split(table, logit: np.ndarray, s1, n: int, out_dir: Path) -> dict:
    """Rows are grouped by S1 (input contract), so run order is first-appearance order; keep S1 runs in that order
    and rows within a run by round-1 rank."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = table.column("s1_id").combine_chunks().dictionary_encode()
    codes = encoded.indices.to_numpy()
    rank = cascade_rank(codes, logit)
    run = np.cumsum(np.r_[0, codes[1:] != codes[:-1]]) if len(codes) else codes
    keep = np.flatnonzero(rank < n)
    keep = keep[np.lexsort((rank[keep], run[keep]))]
    sub = table.take(pa.array(keep))
    out = pa.table({"s1_id": sub.column("s1_id"), "t_id": sub.column("t_id"), "label": sub.column("label"),
                    "filter_score": pa.array(logit[keep].astype(np.float32), pa.float32()),
                    "filter_rank": pa.array(rank[keep].astype(np.int16), pa.int16())})
    for part, start in enumerate(range(0, max(out.num_rows, 1), PART_ROWS)):
        temporary = out_dir / f"part-{part:05d}.parquet.tmp"
        pq.write_table(out.slice(start, PART_ROWS), temporary, compression="zstd")
        temporary.replace(out_dir / f"part-{part:05d}.parquet")
    per_code = np.bincount(codes[keep], minlength=len(encoded.dictionary))
    position = pc.index_in(s1.column("s1_id"), value_set=encoded.dictionary).to_numpy(zero_copy_only=False)
    n_cand = np.where(np.isnan(position.astype(float)), 0, per_code[np.nan_to_num(position.astype(float)).astype(np.int64)])
    s1_out = s1.set_column(s1.schema.get_field_index("n_cand"), "n_cand", pa.array(n_cand.astype(np.int16), pa.int16()))
    pq.write_table(s1_out, out_dir / "s1.parquet", compression="zstd")
    label = out.column("label").to_numpy()
    return {"rows": int(out.num_rows), "s1": int(s1.num_rows), "positives": int((label == 1).sum())}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True, help="round-1 pairs root")
    parser.add_argument("--scores-dir", type=Path, required=True, help="round-1 <ce_out>/scores")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--retention", type=float, default=0.998)
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--n", type=int, default=None, help="override the chosen N")
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    args = parser.parse_args(argv)
    choice = choose_n(args.pairs_root, args.scores_dir, args.retention, args.max_n)
    n = args.n or choice["n"]
    log(f"cascade: top-{n} per S1 by the round-1 cross-encoder (validation retention {choice['curve'][n]:.4%})")
    summary = {"n": n, "choice": choice}
    for split in args.splits:
        table, logit, s1 = read_split(args.pairs_root, args.scores_dir, split)
        summary[split] = write_split(table, logit, s1, n, args.out / split)
        log(f"cascade: {split}: {summary[split]}")
    atomic_write_json(args.out / "cascade_summary.json", summary)


if __name__ == "__main__":
    main()
