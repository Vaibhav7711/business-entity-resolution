"""Join one scored pairs root's cross-encoder logits onto another pairs root's rows, as extra stacker features.

Example: round 1b scored the top-38 of every S1's top-40 list (``pairs_ce1`` + ``ce_r1b/ce/scores``); the round-2
cascade keeps each S1's top-8 by round 1 (``pairs_cascade``). This writes ``<out>/<split>.npz`` aligned with the
cascade rows, for ``ce_stack --extra-dir``: the joined logit (NaN where the source did not score the pair), its rank
within the S1's destination list (descending; unscored rows last) and its gap to the list's best joined logit.
Rows are matched by (S1 id, target id); S1 order comes from the destination's ``s1.parquet``. When the source scored
only its top-K rows on test (``test_k.json`` next to its scores), rows with source filter_rank >= K count as unscored on
every split, so validation features look like test features.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
from pathlib import Path

import numpy as np

from .dense_merge import keys, read_parts, s1_positions, target_codes
from .evaluate_phase1c import atomic_savez, atomic_write_json
from .phase2a_env import log


def join_split(src_root: Path, src_scores: Path, dst_root: Path, split: str,
               max_rank: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Joined logits per destination row (NaN if absent/unscored), destination S1 positions, and match counts."""
    import pyarrow.parquet as pq

    order = pq.read_table(dst_root / split / "s1.parquet", columns=["s1_id"]).column("s1_id").to_pylist()
    position = {s: i for i, s in enumerate(order)}
    src = read_parts(src_root / split, ["s1_id", "t_id", "filter_rank"])
    logits = np.load(src_scores / f"{split}.npy").astype(np.float32)
    if len(logits) != src.num_rows:
        raise ValueError(f"{split}: {len(logits):,} source scores for {src.num_rows:,} source rows")
    if max_rank is not None:
        logits[src.column("filter_rank").to_numpy() >= max_rank] = np.nan
    s_pos = s1_positions(src.column("s1_id"), position)
    s_keys = keys(np.maximum(s_pos, 0), target_codes(src.column("t_id")))
    s_keys[s_pos < 0] = -1
    dst = read_parts(dst_root / split, ["s1_id", "t_id"])
    d_pos = s1_positions(dst.column("s1_id"), position)
    if (d_pos < 0).any():
        raise ValueError(f"{split}: destination rows whose S1 is not in its s1.parquet")
    d_keys = keys(d_pos, target_codes(dst.column("t_id")))
    srt = np.argsort(s_keys, kind="stable")
    ks = s_keys[srt]
    where = np.minimum(np.searchsorted(ks, d_keys), max(len(ks) - 1, 0))
    hit = (ks[where] == d_keys) if len(ks) else np.zeros(len(d_keys), bool)
    joined = np.where(hit, logits[srt][where] if len(ks) else np.nan, np.nan).astype(np.float32)
    info = {"rows": int(len(joined)), "matched": int(hit.sum()), "scored": int(np.isfinite(joined).sum())}
    return joined, d_pos, info


def list_features(joined: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank (0 = best, unscored last) and gap to the group's best finite logit (NaN if unscored or none scored)."""
    filled = np.where(np.isfinite(joined), joined, -np.inf)
    order = np.lexsort((-filled, group))
    starts = np.flatnonzero(np.r_[True, group[order][1:] != group[order][:-1]]) if len(order) else np.zeros(0, np.int64)
    within = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank = np.empty(len(order), np.float32)
    rank[order] = within
    best = np.full(group.max() + 1 if len(group) else 0, -np.inf)
    np.maximum.at(best, group, filled)
    gap = np.where(np.isfinite(joined) & np.isfinite(best[group]), best[group] - joined, np.nan).astype(np.float32)
    return rank, gap


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-root", type=Path, required=True, help="pairs root the source scores are aligned with")
    parser.add_argument("--src-scores", type=Path, required=True, help="directory with <split>.npy logits")
    parser.add_argument("--dst-root", type=Path, required=True, help="pairs root to align the features with")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="r1b")
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    parser.add_argument("--src-max-rank", type=int, help="default: the source's test K (test_k.json), if any")
    args = parser.parse_args(argv)
    from .ce_policy import scored_k

    max_rank = args.src_max_rank if args.src_max_rank is not None else scored_k(args.src_scores)
    log(f"ce_join: source rows with filter_rank >= {max_rank} count as unscored" if max_rank else "ce_join: all source rows")
    names = np.asarray([f"{args.name}_logit", f"{args.name}_rank", f"{args.name}_gap"])
    summary = {}
    for split in args.splits:
        joined, group, info = join_split(args.src_root, args.src_scores, args.dst_root, split, max_rank)
        rank, gap = list_features(joined, group)
        atomic_savez(args.out / f"{split}.npz", {"X": np.column_stack([joined, rank, gap]).astype(np.float32),
                                                 "names": names})
        summary[split] = info | {"src_max_rank": max_rank}
        log(f"ce_join {split}: {info}")
    atomic_write_json(args.out / "join_summary.json", summary)


if __name__ == "__main__":
    main()
