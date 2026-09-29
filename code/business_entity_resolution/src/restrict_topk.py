"""Cascade cut: keep each S1's top-K candidates by a filter-stage score, carrying every row-aligned array along.

The final decision stage then scores (and ``candidate_pairs.tsv`` lists) only these K candidates per S1 instead of the
~51 of the union lists; on validation top-10 by the round-1b cross-encoder keeps 99.93% of the gold links. Rows keep
their original order, so every aligned array stays aligned after the cut:

* the pairs root (``part-*.parquet``; ``s1.parquet`` with ``n_cand`` and, for labelled splits, ``retrieved_truth``
  recomputed; ``truth_len`` untouched, so recall still counts the gold that was cut);
* score directories (``<split>.npy``) and feature directories (``<split>.npz`` with X, names).

    python -m src.restrict_topk --pairs-root A --rank-scores A/scores --k 10 --out B \\
        --scores-dirs A/scores:B/scores --feature-dirs ctx:B_ctx extra:B_extra
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
from pathlib import Path

import numpy as np

from .dense_merge import read_parts, write_parts
from .evaluate_phase1c import atomic_savez, atomic_write_json
from .phase2a_env import log


def keep_mask(s1_codes: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """True for each S1's top-k rows by score (descending; unscored rows last; ties by row order)."""
    filled = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.lexsort((np.arange(len(scores)), -filled, s1_codes))
    grouped = s1_codes[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]]) if len(order) else np.zeros(0, np.int64)
    within = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = np.zeros(len(scores), bool)
    keep[order[within < k]] = True
    return keep


def pairs_spec(text: str) -> tuple[Path, Path]:
    src, _, dst = text.partition(":")
    if not dst:
        raise argparse.ArgumentTypeError(f"{text!r}: expected SRC:DST")
    return Path(src), Path(dst)


def main(argv: list[str] | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--rank-scores", type=Path, required=True, help="<split>.npy filter-stage scores for the cut")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new pairs root")
    parser.add_argument("--scores-dirs", type=pairs_spec, nargs="*", default=[], help="SRC:DST dirs of <split>.npy")
    parser.add_argument("--feature-dirs", type=pairs_spec, nargs="*", default=[], help="SRC:DST dirs of <split>.npz")
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    args = parser.parse_args(argv)
    summary = {"k": args.k}
    for split in args.splits:
        table = read_parts(args.pairs_root / split, ["s1_id", "t_id", "label", "filter_score", "filter_rank"])
        rank_scores = np.load(args.rank_scores / f"{split}.npy")
        if len(rank_scores) != table.num_rows:
            raise ValueError(f"{split}: {len(rank_scores):,} rank scores for {table.num_rows:,} rows")
        s1 = pq.read_table(args.pairs_root / split / "s1.parquet")
        position = {s: i for i, s in enumerate(s1.column("s1_id").to_pylist())}
        encoded = table.column("s1_id").combine_chunks().dictionary_encode()
        lookup = np.fromiter((position[s] for s in encoded.dictionary.to_pylist()), np.int64, len(encoded.dictionary))
        codes = lookup[encoded.indices.to_numpy()]
        keep = keep_mask(codes, rank_scores, args.k)
        cut = table.filter(pa.array(keep))
        write_parts(cut, args.out / split)
        labels = cut.column("label").to_numpy()
        kept_codes = codes[keep]
        s1 = s1.set_column(s1.schema.get_field_index("n_cand"), "n_cand",
                           pa.array(np.bincount(kept_codes, minlength=s1.num_rows).astype(np.int16)))
        if (labels >= 0).all():
            s1 = s1.set_column(s1.schema.get_field_index("retrieved_truth"), "retrieved_truth",
                               pa.array(np.bincount(kept_codes[labels == 1], minlength=s1.num_rows).astype(np.int32)))
        pq.write_table(s1, args.out / split / "s1.parquet", compression="zstd")
        for src, dst in args.scores_dirs:
            x = np.load(src / f"{split}.npy")
            if len(x) != len(keep):
                raise ValueError(f"{src}/{split}.npy: {len(x):,} rows for {len(keep):,}")
            dst.mkdir(parents=True, exist_ok=True)
            np.save(dst / f"{split}.npy", x[keep])
        for src, dst in args.feature_dirs:
            with np.load(src / f"{split}.npz", allow_pickle=False) as z:
                if len(z["X"]) != len(keep):
                    raise ValueError(f"{src}/{split}.npz: {len(z['X']):,} rows for {len(keep):,}")
                atomic_savez(dst / f"{split}.npz", {"X": z["X"][keep], "names": z["names"]})
        summary[split] = {"rows": int(table.num_rows), "kept": int(keep.sum()), "per_s1": float(keep.sum() / s1.num_rows),
                          "gold_kept": float(labels[labels == 1].size / max(int((table.column("label").to_numpy() == 1).sum()), 1))
                          if (labels >= 0).all() else None}
        log(f"restrict_topk {split}: {summary[split]}")
    atomic_write_json(args.out / "restrict_summary.json", summary)


if __name__ == "__main__":
    main()
