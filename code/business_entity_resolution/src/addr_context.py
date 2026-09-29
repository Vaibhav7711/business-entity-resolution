"""Address-text agreement features for the decision stage (street and place words, numbers removed).

A distractor may keep the entity's name and street number but sit on another street ("17 Rue de Bruges" against
"17 Rue Xaintrailles"), a pattern a cross-encoder trained on number-level noise can miss. Per pairs row, with every
digit removed and the address normalised (``normalize_text``), and NaN when either address is empty:

* ``addr_tset``: fuzzy token-set similarity (0-100, rapidfuzz) of the two addresses;
* ``addr_tsort``: fuzzy token-sort similarity (order-insensitive edit similarity);
* ``addr_partial``: fuzzy partial similarity of the shorter address inside the longer one.

Identical definitions on every split and country. Output ``<out>/<split>.npz`` aligned with the pairs rows.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from .dense_merge import read_parts
from .evaluate_phase1c import atomic_savez, atomic_write_json
from .normalization import normalize_text
from .phase2a_env import log

NAMES = ("addr_tset", "addr_tsort", "addr_partial")
DIGITS = re.compile(r"\d+")


def street_text(address: str) -> str:
    return " ".join(DIGITS.sub(" ", normalize_text(address or "")).split())


def read_addresses(paths: list[Path], keep: set) -> dict:
    out = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                if row["entity_id"] in keep:
                    out[row["entity_id"]] = street_text(row["business_address"])
    return out


def pair_features(a: list[str], b: list[str]) -> np.ndarray:
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist

    out = np.column_stack([cpdist(a, b, scorer=s, dtype=np.float32, workers=-1)
                           for s in (fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_ratio)])
    empty = np.fromiter((not x or not y for x, y in zip(a, b)), bool, len(a))
    out[empty] = np.nan
    return out.astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    args = parser.parse_args(argv)
    summary = {}
    for split in args.splits:
        kind, directory = ("test", args.test_dir) if split == "test" else ("train", args.train_dir)
        table = read_parts(args.pairs_root / split, ["s1_id", "t_id"])
        s_enc = table.column("s1_id").combine_chunks().dictionary_encode()
        t_enc = table.column("t_id").combine_chunks().dictionary_encode()
        s_addr = read_addresses([directory / f"{kind}_source1.tsv"], set(s_enc.dictionary.to_pylist()))
        t_addr = read_addresses([directory / f"{kind}_source{k}.tsv" for k in (2, 3)], set(t_enc.dictionary.to_pylist()))
        s_vocab = [s_addr.get(x, "") for x in s_enc.dictionary.to_pylist()]
        t_vocab = [t_addr.get(x, "") for x in t_enc.dictionary.to_pylist()]
        s_codes, t_codes = s_enc.indices.to_numpy(), t_enc.indices.to_numpy()
        X = np.empty((len(s_codes), len(NAMES)), np.float32)
        for start in range(0, len(s_codes), 5_000_000):
            sl = slice(start, start + 5_000_000)
            X[sl] = pair_features([s_vocab[c] for c in s_codes[sl]], [t_vocab[c] for c in t_codes[sl]])
        atomic_savez(args.out / f"{split}.npz", {"X": X, "names": np.asarray(NAMES)})
        summary[split] = {"rows": int(len(X)), "means": np.nanmean(X, axis=0, dtype=np.float64).round(2).tolist(),
                          "nan_share": np.isnan(X).mean(0).round(4).tolist()}
        log(f"addr_context {split}: {summary[split]}")
    atomic_write_json(args.out / "addr_summary.json", summary)


if __name__ == "__main__":
    main()
