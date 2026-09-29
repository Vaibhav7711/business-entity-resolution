"""Corpus-context features for the cross-encoder's decision stage (per pairs row).

The cross-encoder sees one (S1, candidate) text pair; these features give it what only the corpus knows:
how many S1 / targets share the normalised name or address (a unique name makes a name-only target far more
likely to be the S1's; several S1 at one address make a trade-name target there ambiguous), whether the target's
address is empty, exact name/address equality, and how many candidates in the S1's list share the target's name.

Counts are taken on the corpus being searched: training files for fold-0 validation/holdout, test files for test.
The training S1 corpus is subsampled to the test S1 count (all fold-0 S1 plus a seeded sample of the other folds),
so S1 counts have the same scale on validation and test. Names/addresses use the pipeline's ``normalize_text``;
counts are within the record's own country label (an open set).

Output: ``<out>/<split>.npz`` with one float32 column per ``CONTEXT_FEATURES`` entry, aligned with the split's
pairs rows (part files in sorted order).
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
from pathlib import Path

import numpy as np

from .evaluate_phase1c import atomic_savez, atomic_write_json
from .normalization import normalize_text
from .phase2a_env import log

CONTEXT_FEATURES = ("s1_same_name", "s1_same_addr", "tg_same_name", "tg_same_addr", "s1_same_tname",
                    "t_addr_empty", "s1_addr_empty", "name_eq", "addr_eq", "list_same_tname")


def read_records(paths: list[Path]) -> dict:
    ids, country, name, addr = [], [], [], []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                ids.append(row["entity_id"]); country.append(row["country"])
                name.append(normalize_text(row["business_name"])); addr.append(normalize_text(row["business_address"]))
    return {"id": np.asarray(ids, dtype=object), "country": np.asarray(country, dtype=object),
            "name": np.asarray(name, dtype=object), "addr": np.asarray(addr, dtype=object)}


def keys(country: np.ndarray, values: np.ndarray, vocab: dict) -> np.ndarray:
    """Integer code per (country, value); empty values get -1. ``vocab`` is shared so codes compare across corpora."""
    out = np.empty(len(values), np.int64)
    for i, (c, v) in enumerate(zip(country, values)):
        out[i] = -1 if not v else vocab.setdefault((c, v), len(vocab))
    return out


def counts(codes: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(codes[codes >= 0], minlength=size)


def s1_counting_subset(s1_ids: np.ndarray, folds_path: Path | None, target: int | None, seed: int) -> np.ndarray:
    """Boolean mask of S1 used for counting: everything, or all fold-0 S1 + a seeded sample up to ``target``."""
    if folds_path is None or target is None or target >= len(s1_ids):
        return np.ones(len(s1_ids), bool)
    fold = {}
    with folds_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            fold[row["source1_entity_id"]] = int(row["fold"])
    in0 = np.asarray([fold.get(s) == 0 for s in s1_ids])
    rest = np.flatnonzero(~in0)
    rng = np.random.default_rng(seed)
    take = rng.choice(rest, size=max(target - int(in0.sum()), 0), replace=False)
    mask = in0.copy()
    mask[take] = True
    return mask


def split_features(split_dir: Path, s1: dict, tg: dict, s1_mask: np.ndarray) -> np.ndarray:
    import pyarrow.parquet as pq

    vocab_name, vocab_addr = {}, {}
    s1_name = keys(s1["country"], s1["name"], vocab_name); s1_addr = keys(s1["country"], s1["addr"], vocab_addr)
    tg_name = keys(tg["country"], tg["name"], vocab_name); tg_addr = keys(tg["country"], tg["addr"], vocab_addr)
    n_name, n_addr = len(vocab_name), len(vocab_addr)
    s1_name_n, s1_addr_n = counts(s1_name[s1_mask], n_name), counts(s1_addr[s1_mask], n_addr)
    tg_name_n, tg_addr_n = counts(tg_name, n_name), counts(tg_addr, n_addr)
    s1_index = {v: i for i, v in enumerate(s1["id"])}
    tg_index = {v: i for i, v in enumerate(tg["id"])}
    cols = {name: [] for name in CONTEXT_FEATURES}
    for path in sorted(split_dir.glob("part-*.parquet")):
        table = pq.read_table(path, columns=["s1_id", "t_id"])
        s_dict = table.column("s1_id").combine_chunks().dictionary_encode()
        t_dict = table.column("t_id").combine_chunks().dictionary_encode()
        s_rows = np.asarray([s1_index[x] for x in s_dict.dictionary.to_pylist()], np.int64)[s_dict.indices.to_numpy()]
        t_rows = np.asarray([tg_index[x] for x in t_dict.dictionary.to_pylist()], np.int64)[t_dict.indices.to_numpy()]
        sn, sa, tn, ta = s1_name[s_rows], s1_addr[s_rows], tg_name[t_rows], tg_addr[t_rows]
        cols["s1_same_name"].append(np.where(sn >= 0, s1_name_n[np.maximum(sn, 0)], 0))
        cols["s1_same_addr"].append(np.where(sa >= 0, s1_addr_n[np.maximum(sa, 0)], 0))
        cols["tg_same_name"].append(np.where(tn >= 0, tg_name_n[np.maximum(tn, 0)], 0))
        cols["tg_same_addr"].append(np.where(ta >= 0, tg_addr_n[np.maximum(ta, 0)], 0))
        cols["s1_same_tname"].append(np.where(tn >= 0, s1_name_n[np.maximum(tn, 0)], 0))
        cols["t_addr_empty"].append((ta < 0).astype(np.float32))
        cols["s1_addr_empty"].append((sa < 0).astype(np.float32))
        cols["name_eq"].append(((sn == tn) & (sn >= 0)).astype(np.float32))
        cols["addr_eq"].append(((sa == ta) & (sa >= 0)).astype(np.float32))
        cols["list_same_tname"].append(np.stack([s_rows, tn]))        # resolved below (lists may span parts)
    s_all, tn_all = np.concatenate(cols["list_same_tname"], axis=1) if cols["list_same_tname"] else np.zeros((2, 0), np.int64)
    key = s_all * (n_name + 1) + (tn_all + 1)
    _, inverse, per_key = np.unique(key, return_inverse=True, return_counts=True)
    cols["list_same_tname"] = [np.where(tn_all >= 0, per_key[inverse], 0)]
    return np.column_stack([np.concatenate(cols[name]).astype(np.float32) for name in CONTEXT_FEATURES])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True, help="artifacts/folds.tsv (training S1 subsample)")
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    test_s1 = read_records([args.test_dir / "test_source1.tsv"])
    corpora = {}
    summary = {}
    for split in args.splits:
        kind = "test" if split == "test" else "train"
        if kind not in corpora:
            d, prefix = (args.test_dir, "test") if kind == "test" else (args.train_dir, "train")
            s1 = test_s1 if kind == "test" else read_records([d / f"{prefix}_source1.tsv"])
            tg = read_records([d / f"{prefix}_source2.tsv", d / f"{prefix}_source3.tsv"])
            mask = (np.ones(len(s1["id"]), bool) if kind == "test"
                    else s1_counting_subset(s1["id"], args.folds, len(test_s1["id"]), args.seed))
            corpora[kind] = (s1, tg, mask)
            log(f"context: {kind} corpus: {len(s1['id']):,} S1 ({int(mask.sum()):,} counted), {len(tg['id']):,} targets")
        X = split_features(args.pairs_root / split, *corpora[kind])
        atomic_savez(args.out / f"{split}.npz", {"X": X, "names": np.asarray(CONTEXT_FEATURES)})
        summary[split] = {"rows": int(len(X)), "means": dict(zip(CONTEXT_FEATURES, np.round(X.mean(0), 4).tolist()))}
        log(f"context: {split}: {len(X):,} rows")
    atomic_write_json(args.out / "context_summary.json", summary)


if __name__ == "__main__":
    main()
