"""Explicit number and name-token agreement features for the decision stage.

The data's hardest false matches are near-duplicate distractors whose street number differs ("#10125 127th Ave"
against the true "11311 127th Ave") or whose name swaps one token ("Lyrium Raoyal" against "Lyrium Roman"). A text
cross-encoder compares digits weakly, so per pairs row this adds (NaN where a side has no number / no name token):

* ``num_first_eq``: the first number of the target's address equals the S1's first address number;
* ``num_shared`` / ``num_jaccard``: numbers shared by both addresses (count, Jaccard of the first ``MAX_NUMS``);
* ``num_t_only``: numbers in the target's address that the S1's address lacks;
* ``name_tok_jaccard``: Jaccard of the normalised name tokens (first ``MAX_TOKS``);
* ``name_first_eq``: the first name tokens are equal.

Each entity's numbers and tokens are hashed once; row comparisons are vectorised. Identical definitions on every
split. Output ``<out>/<split>.npz`` aligned with the pairs rows.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import re
import zlib
from pathlib import Path

import numpy as np

from .dense_merge import read_parts
from .evaluate_phase1c import atomic_savez, atomic_write_json
from .normalization import normalize_text
from .phase2a_env import log

MAX_NUMS, MAX_TOKS = 4, 6
NAMES = ("num_first_eq", "num_shared", "num_jaccard", "num_t_only", "name_tok_jaccard", "name_first_eq")
NUMBER = re.compile(r"\d+")
LEGAL = {"inc", "llc", "ltd", "co", "corp", "limited", "private", "pvt", "the", "and", "of", "company"}


def number_codes(address: str) -> list[int]:
    """Distinct numbers of an address in order of appearance (leading zeros dropped), as ints (capped)."""
    seen: list[int] = []
    for token in NUMBER.findall(address or ""):
        value = int(token.lstrip("0") or "0") if len(token) <= 12 else zlib.crc32(token.encode())
        if value not in seen:
            seen.append(value)
        if len(seen) == MAX_NUMS:
            break
    return seen


def token_codes(name: str) -> list[int]:
    """Distinct normalised name tokens (legal suffixes dropped) as crc32 codes, in order."""
    seen: list[int] = []
    for token in normalize_text(name or "").split():
        if token in LEGAL:
            continue
        code = zlib.crc32(token.encode())
        if code not in seen:
            seen.append(code)
        if len(seen) == MAX_TOKS:
            break
    return seen


def pad(rows: list[list[int]], width: int) -> np.ndarray:
    out = np.full((len(rows), width), -1, np.int64)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def entity_table(paths: list[Path], keep: set) -> tuple[dict, np.ndarray, np.ndarray]:
    ids, nums, toks = [], [], []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                if row["entity_id"] in keep:
                    ids.append(row["entity_id"])
                    nums.append(number_codes(row["business_address"]))
                    toks.append(token_codes(row["business_name"]))
    return {e: i for i, e in enumerate(ids)}, pad(nums, MAX_NUMS), pad(toks, MAX_TOKS)


def set_stats(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: |a|, |b|, |a & b| for padded (-1) sets of distinct values."""
    na, nb = (a >= 0).sum(1), (b >= 0).sum(1)
    inter = np.zeros(len(a), np.int64)
    for j in range(a.shape[1]):
        inter += ((a[:, [j]] == b) & (a[:, [j]] >= 0)).any(1)
    return na, nb, inter


def row_features(s_nums, s_toks, t_nums, t_toks) -> np.ndarray:
    na, nb, shared = set_stats(t_nums, s_nums)
    union = na + nb - shared
    both = (na > 0) & (nb > 0)
    first_eq = np.where(both, (t_nums[:, 0] == s_nums[:, 0]).astype(np.float32), np.nan)
    jac = np.where(union > 0, shared / np.maximum(union, 1), np.nan)
    t_only = np.where(na > 0, (na - shared).astype(np.float32), np.nan)
    ta, sb, tshared = set_stats(t_toks, s_toks)
    tunion = ta + sb - tshared
    tok_jac = np.where(tunion > 0, tshared / np.maximum(tunion, 1), np.nan)
    tok_first = np.where((ta > 0) & (sb > 0), (t_toks[:, 0] == s_toks[:, 0]).astype(np.float32), np.nan)
    return np.column_stack([first_eq, np.where(both, shared, np.nan), jac, t_only, tok_jac, tok_first]).astype(np.float32)


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
        s_ix, s_nums, s_toks = entity_table([directory / f"{kind}_source1.tsv"], set(s_enc.dictionary.to_pylist()))
        t_ix, t_nums, t_toks = entity_table([directory / f"{kind}_source{k}.tsv" for k in (2, 3)],
                                            set(t_enc.dictionary.to_pylist()))
        s_rows = np.asarray([s_ix[x] for x in s_enc.dictionary.to_pylist()], np.int64)[s_enc.indices.to_numpy()]
        t_rows = np.asarray([t_ix[x] for x in t_enc.dictionary.to_pylist()], np.int64)[t_enc.indices.to_numpy()]
        X = np.empty((len(s_rows), len(NAMES)), np.float32)
        for start in range(0, len(s_rows), 5_000_000):                     # bounded memory on the test split
            sl = slice(start, start + 5_000_000)
            X[sl] = row_features(s_nums[s_rows[sl]], s_toks[s_rows[sl]], t_nums[t_rows[sl]], t_toks[t_rows[sl]])
        atomic_savez(args.out / f"{split}.npz", {"X": X, "names": np.asarray(NAMES)})
        summary[split] = {"rows": int(len(X)), "means": np.nanmean(X, axis=0, dtype=np.float64).round(4).tolist(),
                          "nan_share": np.isnan(X).mean(0).round(4).tolist()}
        log(f"num_context {split}: {summary[split]}")
    atomic_write_json(args.out / "num_summary.json", summary)


if __name__ == "__main__":
    main()
