"""Merge the dense-retrieval route into the candidate lists.

Inputs: the top-40 pairs root (``topk_export`` format) and the dense top-K lists (``bi_encoder``:
``<dense>/<split>/part-*.parquet`` with s1_id, t_id, dense_score, dense_rank).

``--stage new`` writes ``<new>/<split>``: only the dense pairs (rank < ``--max-rank``) that are not already in the S1's
top-40 list, in the pairs format (filter_score = dense_score, filter_rank = dense_rank) so ``ce_model`` can score them.
Labels for fold-0 splits come from the training ground truth; test labels are -1.

``--stage augment`` (after the new pairs are scored) writes ``<aug>/<split>``: for every S1, its top-40 rows followed by
its new rows (filter_rank = 40 + dense_rank, filter_score = -20), ``s1.parquet`` with updated counts,
``<aug>/scores/<split>.npy`` (base and new cross-encoder logits in augmented row order) and ``<aug>/extra/<split>.npz``
(dense_score, dense_rank, dense_only) for the stacker. Candidate lists only grow, so recall only rises. Base rows the
base cross-encoder did not score on test (filter_rank >= its test K, ``test_k.json`` next to ``--base-scores``) are
dropped on every split, so the augmented lists hold only scored rows and need no K of their own.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluate_phase1c import atomic_savez, atomic_write_json
from .phase2a_env import log

BASE_K = 40
EXTRA_FEATURES = ("dense_score", "dense_rank", "dense_only")


def read_parts(split_dir: Path, columns: list[str]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa.concat_tables([pq.read_table(p, columns=columns) for p in sorted(split_dir.glob("part-*.parquet"))])


def write_parts(table, out_dir: Path, rows: int = 4_000_000) -> None:
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    for part, start in enumerate(range(0, max(table.num_rows, 1), rows)):
        temporary = out_dir / f"part-{part:05d}.parquet.tmp"
        pq.write_table(table.slice(start, rows), temporary, compression="zstd")
        temporary.replace(out_dir / f"part-{part:05d}.parquet")


def load_gold(train_dir: Path, wanted: set) -> dict:
    gold = {}
    with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if row["source1_entity_id"] in wanted and row["matched_entity_ids"]:
                gold[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))
    return gold


def s1_positions(column, position: dict) -> np.ndarray:
    """Position of each row's S1 in the split's s1.parquet order (-1 if absent), via the column's dictionary."""
    encoded = column.combine_chunks().dictionary_encode()
    lookup = np.fromiter((position.get(s, -1) for s in encoded.dictionary.to_pylist()), np.int64, len(encoded.dictionary))
    return lookup[encoded.indices.to_numpy()]


def target_codes(column) -> np.ndarray:
    from .blocking import encode_id

    encoded = column.combine_chunks().dictionary_encode()
    lookup = np.fromiter((encode_id(t) for t in encoded.dictionary.to_pylist()), np.int64, len(encoded.dictionary))
    return lookup[encoded.indices.to_numpy()]


def keys(s1_pos: np.ndarray, t_code: np.ndarray) -> np.ndarray:
    return (s1_pos << 33) | t_code


def new_pairs(base_root: Path, dense_root: Path, split: str, max_rank: int, gold: dict | None, out: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    s1_order = pq.read_table(base_root / split / "s1.parquet", columns=["s1_id"]).column("s1_id").to_pylist()
    position = {s: i for i, s in enumerate(s1_order)}
    base = read_parts(base_root / split, ["s1_id", "t_id"])
    base_keys = np.unique(keys(s1_positions(base.column("s1_id"), position), target_codes(base.column("t_id"))))
    del base
    dense = read_parts(dense_root / split, ["s1_id", "t_id", "dense_score", "dense_rank"])
    d_pos = s1_positions(dense.column("s1_id"), position)
    d_rank = dense.column("dense_rank").to_numpy().astype(np.int64)
    d_keys = keys(np.maximum(d_pos, 0), target_codes(dense.column("t_id")))
    keep = np.flatnonzero((d_rank < max_rank) & (d_pos >= 0) & ~np.isin(d_keys, base_keys))
    keep = keep[np.lexsort((d_rank[keep], d_pos[keep]))]
    sub = dense.take(pa.array(keep))
    s1, t = sub.column("s1_id").to_pylist(), sub.column("t_id").to_pylist()
    label = [(1 if tt in gold.get(ss, ()) else 0) if gold is not None else -1 for ss, tt in zip(s1, t)]
    table = pa.table({"s1_id": sub.column("s1_id").cast(pa.string()), "t_id": sub.column("t_id").cast(pa.string()),
                      "label": pa.array(label, pa.int8()), "filter_score": sub.column("dense_score").cast(pa.float32()),
                      "filter_rank": sub.column("dense_rank").cast(pa.int16())})
    write_parts(table, out / split)
    base_s1 = pq.read_table(base_root / split / "s1.parquet")
    n_cand = np.bincount(d_pos[keep], minlength=len(s1_order))
    pq.write_table(base_s1.set_column(base_s1.schema.get_field_index("n_cand"), "n_cand", pa.array(n_cand.astype(np.int16))),
                   out / split / "s1.parquet")
    info = {"rows": int(len(keep)), "positives": int(sum(x == 1 for x in label)), "s1_with_new": int((n_cand > 0).sum())}
    log(f"dense_merge new {split}: {info}")
    return info


def augment(base_root: Path, new_root: Path, dense_root: Path, split: str, base_scores: Path, new_scores: Path,
            out: Path, base_max_rank: int | None = None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = ["s1_id", "t_id", "label", "filter_score", "filter_rank"]
    base, new = read_parts(base_root / split, cols), read_parts(new_root / split, cols)
    b_scores = np.load(base_scores / f"{split}.npy")
    if len(b_scores) != base.num_rows:
        raise ValueError(f"{split}: base score rows do not match base pairs rows")
    if base_max_rank is not None:
        scored = base.column("filter_rank").to_numpy() < base_max_rank
        base, b_scores = base.filter(pa.array(scored)), b_scores[scored]
    s1 = pq.read_table(base_root / split / "s1.parquet")
    position = {s: i for i, s in enumerate(s1.column("s1_id").to_pylist())}
    b_pos, n_pos = s1_positions(base.column("s1_id"), position), s1_positions(new.column("s1_id"), position)
    order = np.lexsort((np.r_[np.arange(base.num_rows), np.arange(new.num_rows)],
                        np.r_[np.zeros(base.num_rows, np.int8), np.ones(new.num_rows, np.int8)], np.r_[b_pos, n_pos]))
    new_adj = pa.table({"s1_id": new.column("s1_id").cast(pa.string()), "t_id": new.column("t_id").cast(pa.string()),
                        "label": new.column("label").cast(pa.int8()),
                        "filter_score": pa.array(np.full(new.num_rows, -20.0, np.float32)),
                        "filter_rank": pa.array((BASE_K + new.column("filter_rank").to_numpy()).astype(np.int16))})
    merged = pa.concat_tables([base.cast(new_adj.schema), new_adj]).take(pa.array(order))
    del base, new
    write_parts(merged, out / split)
    n_scores = np.load(new_scores / f"{split}.npy")
    if len(n_scores) != len(n_pos):
        raise ValueError(f"{split}: new score rows do not match new pairs rows")
    if not (np.isfinite(b_scores).all() and np.isfinite(n_scores).all()):
        raise ValueError(f"{split}: unscored rows left in the augmented lists")
    (out / "scores").mkdir(parents=True, exist_ok=True)
    np.save(out / "scores" / f"{split}.npy", np.r_[b_scores, n_scores][order].astype(np.float32))
    m_pos = np.r_[b_pos, n_pos][order]
    m_keys = keys(m_pos, target_codes(merged.column("t_id")))
    dense = read_parts(dense_root / split, ["s1_id", "t_id", "dense_score", "dense_rank"])
    d_pos = s1_positions(dense.column("s1_id"), position)
    d_keys = keys(np.maximum(d_pos, 0), target_codes(dense.column("t_id")))
    d_keys[d_pos < 0] = -1
    srt = np.argsort(d_keys, kind="stable"); ks = d_keys[srt]
    where = np.minimum(np.searchsorted(ks, m_keys), len(ks) - 1)
    hit = ks[where] == m_keys
    d_score = dense.column("dense_score").to_numpy().astype(np.float32)[srt][where]
    d_rank = dense.column("dense_rank").to_numpy().astype(np.float32)[srt][where]
    dense_only = np.r_[np.zeros(len(b_pos)), np.ones(len(n_pos))][order]
    extra = np.column_stack([np.where(hit, d_score, np.nan), np.where(hit, d_rank, 999.0), dense_only]).astype(np.float32)
    atomic_savez(out / "extra" / f"{split}.npz", {"X": extra, "names": np.asarray(EXTRA_FEATURES)})
    labels = merged.column("label").to_numpy()
    s1 = s1.set_column(s1.schema.get_field_index("n_cand"), "n_cand",
                       pa.array(np.bincount(m_pos, minlength=s1.num_rows).astype(np.int16)))
    if (labels >= 0).all():
        s1 = s1.set_column(s1.schema.get_field_index("retrieved_truth"), "retrieved_truth",
                           pa.array(np.bincount(m_pos[labels == 1], minlength=s1.num_rows).astype(np.int32)))
    pq.write_table(s1, out / split / "s1.parquet", compression="zstd")
    info = {"rows": int(merged.num_rows), "new_rows": int(len(n_pos)), "positives": int((labels == 1).sum()),
            "dense_hits_in_list": int(hit.sum())}
    log(f"dense_merge augment {split}: {info}")
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("new", "augment"), required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--aug-root", type=Path)
    parser.add_argument("--base-scores", type=Path)
    parser.add_argument("--new-scores", type=Path)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    args = parser.parse_args(argv)
    summary = {}
    for split in args.splits:
        if args.stage == "new":
            gold = None
            if split != "test":
                import pyarrow.parquet as pq

                ids = set(pq.read_table(args.base_root / split / "s1.parquet", columns=["s1_id"]).column("s1_id").to_pylist())
                gold = load_gold(args.train_dir, ids)
            summary[split] = new_pairs(args.base_root, args.dense_root, split, args.max_rank, gold, args.new_root)
        else:
            from .ce_policy import scored_k

            summary[split] = augment(args.base_root, args.new_root, args.dense_root, split, args.base_scores,
                                     args.new_scores, args.aug_root, scored_k(args.base_scores))
    target = args.new_root if args.stage == "new" else args.aug_root
    atomic_write_json(target / f"dense_merge_{args.stage}.json", summary)


if __name__ == "__main__":
    main()
