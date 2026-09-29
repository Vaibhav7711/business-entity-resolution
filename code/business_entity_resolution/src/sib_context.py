"""Sibling name-support features for the decision stage.

A name-only candidate (empty address) is judged on its name alone. Besides the S1's own name, the S1's other strong
candidates are independent noisy copies of the same entity's name, so a true duplicate tends to resemble them while
a near-duplicate distractor ("Lyrium Raoyal" next to "Lyrium Roman") does not. Per pairs row:

* ``sib_max_sim`` / ``sib_mean_sim``: fuzzy token-set similarity (0-100, rapidfuzz) of the candidate's normalised
  name to the names of the S1's top ``TOP`` candidates by cross-encoder logit, excluding the row itself (NaN if none);
* ``name_sim_s1``: fuzzy token-set similarity of the candidate's name to the S1's name.

The ranking uses the pairs root's scores (the same model on every split), so the features mean the same on
validation, holdout and test. Output ``<out>/<split>.npz`` aligned with the pairs rows.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
from pathlib import Path

import numpy as np

from .dense_merge import read_parts
from .evaluate_phase1c import atomic_savez, atomic_write_json
from .normalization import normalize_text
from .phase2a_env import log

TOP = 3
NAMES = ("sib_max_sim", "sib_mean_sim", "name_sim_s1")


def read_names(paths: list[Path], keep: set) -> dict:
    names = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                if row["entity_id"] in keep:
                    names[row["entity_id"]] = normalize_text(row["business_name"] or "")
    return names


def group_features(t_names: list[str], logits: np.ndarray, top: int = TOP) -> np.ndarray:
    """(n, 2) sibling max/mean similarity for one S1's rows (in row order): the first ``top`` of the S1's candidates
    by logit, skipping the row itself."""
    import warnings

    from rapidfuzz import fuzz
    from rapidfuzz.process import cdist

    order = np.argsort(-np.where(np.isfinite(logits), logits, -np.inf), kind="stable")[:top + 1]
    sims = cdist(t_names, [t_names[j] for j in order], scorer=fuzz.token_set_ratio, dtype=np.float32)
    sims[order, np.arange(len(order))] = np.nan                          # a row is not its own sibling
    head = sims[:, :top]
    self_in_head = np.isnan(head).any(axis=1)                            # then the (top+1)-th fills its place
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx = np.where(self_in_head, np.nanmax(sims, axis=1), np.nanmax(head, axis=1) if head.size else np.nan)
        mn = np.where(self_in_head, np.nanmean(sims, axis=1), np.nanmean(head, axis=1) if head.size else np.nan)
    return np.column_stack([mx, mn]).astype(np.float32)


def split_features(split_dir: Path, scores: np.ndarray, s1_names: dict, t_names: dict) -> np.ndarray:
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist

    table = read_parts(split_dir, ["s1_id", "t_id"])
    if len(scores) != table.num_rows:
        raise ValueError(f"{split_dir.name}: {len(scores):,} scores for {table.num_rows:,} rows")
    s_enc = table.column("s1_id").combine_chunks().dictionary_encode()
    t_enc = table.column("t_id").combine_chunks().dictionary_encode()
    s_codes, t_codes = s_enc.indices.to_numpy(), t_enc.indices.to_numpy()
    s_vocab = [s1_names.get(x, "") for x in s_enc.dictionary.to_pylist()]
    t_vocab = [t_names.get(x, "") for x in t_enc.dictionary.to_pylist()]
    tn = [t_vocab[c] for c in t_codes]
    out = np.full((len(tn), len(NAMES)), np.nan, np.float32)
    out[:, 2] = cpdist(tn, [s_vocab[c] for c in s_codes], scorer=fuzz.token_set_ratio, dtype=np.float32, workers=-1)
    bounds = np.r_[np.flatnonzero(np.r_[True, s_codes[1:] != s_codes[:-1]]), len(tn)] if len(tn) else np.zeros(1, np.int64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[a:b, :2] = group_features(tn[a:b], scores[a:b])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    args = parser.parse_args(argv)
    summary = {}
    for split in args.splits:
        kind, directory = ("test", args.test_dir) if split == "test" else ("train", args.train_dir)
        table = read_parts(args.pairs_root / split, ["s1_id", "t_id"])
        import pyarrow.compute as pc

        s1_keep = set(pc.unique(table.column("s1_id")).to_pylist())
        t_keep = set(pc.unique(table.column("t_id")).to_pylist())
        s1_names = read_names([directory / f"{kind}_source1.tsv"], s1_keep)
        t_names = read_names([directory / f"{kind}_source{k}.tsv" for k in (2, 3)], t_keep)
        X = split_features(args.pairs_root / split, np.load(args.scores_dir / f"{split}.npy"), s1_names, t_names)
        atomic_savez(args.out / f"{split}.npz", {"X": X, "names": np.asarray(NAMES)})
        summary[split] = {"rows": int(len(X)), "means": np.nanmean(X, axis=0, dtype=np.float64).round(2).tolist()}
        log(f"sib_context {split}: {summary[split]}")
    atomic_write_json(args.out / "sib_summary.json", summary)


if __name__ == "__main__":
    main()
