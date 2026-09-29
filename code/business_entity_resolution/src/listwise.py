"""Listwise (group-relative) features of the candidate scores within each S1's list, for the decision stage.

For each score column (the cross-encoder logit, the bge-reranker logit, the dense cosine score) and each S1 list
(at most ``WIDTH`` rows): group count, min, max, mean, median, std, 25th/75th percentile; per row its z-score,
robust z-score ((s - median) / IQR), gaps to the list's top-1/2/3, the number of rows within fixed deltas of the
top-1, its softmax share and the list's softmax entropy. Plus cross-model rank disagreement (absolute rank
differences between the three scores). NaN scores (e.g. the reranker outside its top 8) are left out of a group.
Output ``<out>/<split>.npz`` aligned with the pairs rows.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import warnings
from pathlib import Path

import numpy as np

from .dense_merge import read_parts
from .evaluate_phase1c import atomic_savez, atomic_write_json
from .phase2a_env import log

WIDTH = 10
DELTAS = {"ce": (0.5, 1.0, 2.0), "rr": (0.5, 1.0, 2.0), "dense": (0.02, 0.05, 0.1)}


def group_matrix(codes: np.ndarray, values: np.ndarray, width: int = WIDTH):
    """(n_groups, width) padded matrix (NaN), each row's (group, slot), in original row order."""
    order = np.lexsort((np.arange(len(codes)), codes))
    gs = codes[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]) if len(order) else np.zeros(0, np.int64)
    sizes = np.diff(np.r_[starts, len(order)])
    if len(sizes) and sizes.max() > width:
        raise ValueError(f"a list has {sizes.max()} rows; width is {width}")
    group = np.repeat(np.arange(len(starts)), sizes)
    slot = np.arange(len(order)) - np.repeat(starts, sizes)
    M = np.full((len(starts), width), np.nan)
    M[group, slot] = values[order]
    row_group, row_slot = np.empty(len(codes), np.int64), np.empty(len(codes), np.int64)
    row_group[order], row_slot[order] = group, slot
    return M, row_group, row_slot


def score_features(name: str, M: np.ndarray, row_group: np.ndarray, s: np.ndarray) -> tuple[list, list]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cnt = np.isfinite(M).sum(1).astype(float)
        mn, mx, mean = np.nanmin(M, 1), np.nanmax(M, 1), np.nanmean(M, 1)
        med, sd = np.nanmedian(M, 1), np.nanstd(M, 1)
        q25, q75 = np.nanpercentile(M, 25, axis=1), np.nanpercentile(M, 75, axis=1)
        top = -np.sort(-np.where(np.isfinite(M), M, -np.inf), axis=1)[:, :3]
        top = np.where(np.isfinite(top), top, np.nan)
        ex = np.exp(M - mx[:, None])
        z_sum = np.nansum(ex, 1)
        share_m = ex / z_sum[:, None]
        entropy = -np.nansum(np.where(share_m > 0, share_m * np.log(share_m), 0.0), 1)
        g = row_group
        feats = [cnt[g], mn[g], mx[g], mean[g], med[g], sd[g], q25[g], q75[g],
                 (s - mean[g]) / np.where(sd[g] > 0, sd[g], np.nan),
                 (s - med[g]) / np.where((q75 - q25)[g] > 0, (q75 - q25)[g], np.nan),
                 top[g, 0] - s, top[g, 1] - s, top[g, 2] - s]
        names = [f"{name}_{k}" for k in ("cnt", "min", "max", "mean", "median", "std", "q25", "q75", "z", "rz",
                                          "gap1", "gap2", "gap3")]
        for d in DELTAS[name]:
            within = (np.isfinite(M) & (M >= (mx - d)[:, None])).sum(1).astype(float)
            feats.append(within[g]); names.append(f"{name}_within{d}")
        feats += [np.exp(s - mx[g]) / z_sum[g], entropy[g]]
        names += [f"{name}_softmax", f"{name}_entropy"]
    return feats, names


def ranks(M: np.ndarray, row_group: np.ndarray, row_slot: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(M), M, -np.inf)
    order = np.argsort(-filled, axis=1, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(M.shape[1])[None, :].repeat(M.shape[0], 0), axis=1)
    out = r[row_group, row_slot].astype(float)
    out[~np.isfinite(M[row_group, row_slot])] = np.nan
    return out


def build(codes: np.ndarray, scores: dict) -> tuple[np.ndarray, list]:
    cols, names, rk = [], [], {}
    for name, s in scores.items():
        M, rg, rs = group_matrix(codes, s.astype(float))
        f, n = score_features(name, M, rg, s.astype(float))
        cols += f; names += n
        rk[name] = ranks(M, rg, rs)
    keys = list(rk)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cols.append(np.abs(rk[keys[i]] - rk[keys[j]])); names.append(f"rankdiff_{keys[i]}_{keys[j]}")
    return np.column_stack(cols).astype(np.float32), names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True, help="cross-encoder logits (<split>.npy)")
    parser.add_argument("--rr-dir", type=Path, required=True, help="ce_join features with rr_logit first")
    parser.add_argument("--extra-dir", type=Path, required=True, help="dense_merge features with dense_score first")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    args = parser.parse_args(argv)
    summary = {}
    for split in args.splits:
        table = read_parts(args.pairs_root / split, ["s1_id"])
        codes = table.column("s1_id").combine_chunks().dictionary_encode().indices.to_numpy().astype(np.int64)
        with np.load(args.rr_dir / f"{split}.npz") as z:
            rr = z["X"][:, list(z["names"]).index("rr_logit")]
        with np.load(args.extra_dir / f"{split}.npz") as z:
            dense = z["X"][:, list(z["names"]).index("dense_score")]
        X, names = build(codes, {"ce": np.load(args.scores_dir / f"{split}.npy"), "rr": rr, "dense": dense})
        atomic_savez(args.out / f"{split}.npz", {"X": X, "names": np.asarray(names)})
        summary[split] = {"rows": int(len(X)), "features": len(names)}
        log(f"listwise {split}: {summary[split]}")
    atomic_write_json(args.out / "listwise_summary.json", summary)


if __name__ == "__main__":
    main()
