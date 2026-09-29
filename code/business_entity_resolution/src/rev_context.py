"""Target-side competition features from the dense route, for the decision stage.

A candidate target with no address can only be judged by its name; whether this S1 is its owner depends on whether
any *other* S1 is about as similar. For each (S1, target) row this computes, over a competitor corpus of dense lists
(S1 -> top-K targets with cosine scores):

* ``rev_best_other``: the best dense score of the target among the corpus's other S1 (NaN if none lists it);
* ``rev_margin``: the row's own dense score minus that (NaN if either is unknown);
* ``rev_count_other``: how many other corpus S1 list the target;
* ``rev_is_top``: 1 if no other corpus S1 scores the target higher than this row's S1 (NaN if the own score is unknown).

The own score is the row's ``dense_score`` (from ``dense_merge`` extras). Corpus scale is matched across splits: the
validation/holdout corpus is every fold-0 S1 (validation, holdout and the fold-0 training S1; the bi-encoder never
trained on fold 0), i.e. a fraction f of all training S1; the test corpus is a seeded sample of the same fraction f of
the test S1. Output ``<out>/<split>.npz`` aligned with the pairs rows.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
from pathlib import Path

import numpy as np

from .dense_merge import read_parts, target_codes
from .evaluate_phase1c import atomic_savez, atomic_write_json
from .phase2a_env import log

NAMES = ("rev_best_other", "rev_margin", "rev_count_other", "rev_is_top")


def s1_codes(column, index: dict) -> np.ndarray:
    """Integer code per S1 id (added to ``index`` on first sight), via the column's dictionary."""
    encoded = column.combine_chunks().dictionary_encode()
    lookup = np.fromiter((index.setdefault(s, len(index)) for s in encoded.dictionary.to_pylist()), np.int64,
                         len(encoded.dictionary))
    return lookup[encoded.indices.to_numpy()]


def corpus_table(dense_dirs: list[Path], keep_s1: set | None, index: dict, top_k: int) -> tuple[np.ndarray, ...]:
    """(s1 code, target code, score) of every corpus dense row with rank < top_k (and S1 in keep_s1 if given)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    s, t, v = [], [], []
    for directory in dense_dirs:
        table = read_parts(directory, ["s1_id", "t_id", "dense_score", "dense_rank"])
        mask = pc.less(table.column("dense_rank"), top_k)
        if keep_s1 is not None:
            mask = pc.and_(mask, pc.is_in(table.column("s1_id"), value_set=pa.array(sorted(keep_s1), pa.string())))
        table = table.filter(mask)
        s.append(s1_codes(table.column("s1_id"), index))
        t.append(target_codes(table.column("t_id")))
        v.append(table.column("dense_score").to_numpy().astype(np.float32))
    return np.concatenate(s), np.concatenate(t), np.concatenate(v)


def top_two(c_s1: np.ndarray, c_t: np.ndarray, c_v: np.ndarray):
    """Per distinct target: its code, best (s1, score), second best score, and the number of corpus S1 listing it."""
    order = np.lexsort((-c_v, c_t))
    t, s, v = c_t[order], c_s1[order], c_v[order]
    start = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    count = np.diff(np.r_[start, len(t)])
    second = np.where(count > 1, v[np.minimum(start + 1, len(v) - 1)], np.nan)
    return t[start], s[start], v[start], second.astype(np.float32), count


def in_sorted(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    where = np.minimum(np.searchsorted(sorted_keys, keys), max(len(sorted_keys) - 1, 0))
    return (sorted_keys[where] == keys) if len(sorted_keys) else np.zeros(len(keys), bool)


def features(row_s1: np.ndarray, row_t: np.ndarray, own: np.ndarray, corpus, corpus_keys: np.ndarray) -> np.ndarray:
    """``corpus`` from top_two; ``corpus_keys`` the sorted (s1 << 33 | target) keys of the corpus rows."""
    targets, best_s1, best, second, count = corpus
    hit = in_sorted(targets, row_t)
    where = np.minimum(np.searchsorted(targets, row_t), max(len(targets) - 1, 0))
    self_best = hit & (best_s1[where] == row_s1)
    best_other = np.where(hit, np.where(self_best, second[where], best[where]), np.nan).astype(np.float32)
    listed_self = in_sorted(corpus_keys, (row_s1 << 33) | row_t)
    count_other = np.where(hit, count[where] - listed_self, 0).astype(np.float32)
    margin = np.where(np.isfinite(own) & np.isfinite(best_other), own - best_other, np.nan).astype(np.float32)
    is_top = np.where(np.isfinite(own), (~np.isfinite(best_other) | (own >= best_other)).astype(np.float32), np.nan)
    return np.column_stack([best_other, margin, count_other, is_top.astype(np.float32)])


def fold0_fraction(folds_path: Path) -> tuple[int, int]:
    n0 = total = 0
    with folds_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            total += 1
            n0 += int(row["fold"]) == 0
    return n0, total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True, help="augmented lists (pairs_aug)")
    parser.add_argument("--extra-dir", type=Path, required=True, help="dense_merge extras (dense_score first column)")
    parser.add_argument("--train-corpus", type=Path, nargs="+", required=True,
                        help="dense list dirs of every fold-0 S1 (validation, holdout, fold-0 train)")
    parser.add_argument("--test-corpus", type=Path, required=True, help="dense list dir of all test S1")
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    args = parser.parse_args(argv)
    n0, total = fold0_fraction(args.folds)
    fraction = n0 / total
    summary = {"fold0_s1": n0, "train_s1": total, "fraction": fraction}
    corpora: dict[str, tuple] = {}
    index: dict = {}
    for split in args.splits:
        kind = "test" if split == "test" else "train"
        if kind not in corpora:
            if kind == "train":
                c = corpus_table(args.train_corpus, None, index, args.top_k)
            else:
                import pyarrow.compute as pc

                all_s1 = sorted(pc.unique(read_parts(args.test_corpus, ["s1_id"]).column("s1_id")).to_pylist())
                rng = np.random.default_rng(args.seed)
                keep = set(np.asarray(all_s1, dtype=object)[rng.random(len(all_s1)) < fraction].tolist())
                summary["test_corpus_s1"] = len(keep)
                c = corpus_table([args.test_corpus], keep, index, args.top_k)
            corpora[kind] = (top_two(*c), np.unique((c[0] << 33) | c[1]))
            log(f"rev_context: {kind} corpus {len(c[0]):,} dense rows, {len(corpora[kind][0][0]):,} targets")
        rows = read_parts(args.pairs_root / split, ["s1_id", "t_id"])
        with np.load(args.extra_dir / f"{split}.npz", allow_pickle=False) as z:
            names = [str(x) for x in z["names"]]
            own = z["X"][:, names.index("dense_score")].astype(np.float32)
        if len(own) != rows.num_rows:
            raise ValueError(f"{split}: {len(own):,} extra rows for {rows.num_rows:,} pairs rows")
        X = features(s1_codes(rows.column("s1_id"), index), target_codes(rows.column("t_id")), own, *corpora[kind])
        atomic_savez(args.out / f"{split}.npz", {"X": X, "names": np.asarray(NAMES)})
        summary[split] = {"rows": int(len(X)), "listed_by_other": float(np.mean(X[:, 2] > 0)),
                          "is_top": float(np.nanmean(X[:, 3])), "margin_mean": float(np.nanmean(X[:, 1]))}
        log(f"rev_context {split}: {summary[split]}")
    atomic_write_json(args.out / "rev_summary.json", summary)


if __name__ == "__main__":
    main()
