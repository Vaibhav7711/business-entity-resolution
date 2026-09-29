"""Sibling-consensus and raw-quirk features for the decision stage (computed on the final candidate lists).

True copies of an entity share raw quirks (typos, spacing, casing, unit numbers) that ``normalize_text`` erases, and
they sit on the same street as the entity's other copies, while distractors keep the name but move to another street
or share only the address. Siblings of a row are the S1's top ``TOP`` other candidates by the cross-encoder logit
(the same model on every split). Per pairs row (NaN when a side is empty or no sibling qualifies):

* ``raw_name_ratio_s1`` / ``raw_name_eq_s1``: fuzzy edit similarity / exact equality of the raw names (case and
  punctuation kept) of the candidate and the S1;
* ``raw_name_ratio_sib_max`` / ``raw_name_eq_sib``: best raw-name similarity to a sibling / number of siblings with
  the identical raw name;
* ``addr_raw_ratio_s1``: fuzzy edit similarity of the raw addresses (numbers kept);
* ``street_sim_s1_exp``: token-sort similarity of the digit-free street text with generic abbreviation expansion;
* ``street_sim_sib_max`` / ``street_sim_sib_mean``: token-set similarity of that street text to the siblings';
* ``first_num_agree_sib``: share of the siblings with an address number whose first number equals the candidate's.

Output ``<out>/<split>.npz`` aligned with the pairs rows.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import re
import unicodedata
from pathlib import Path

import numpy as np

from .dense_merge import read_parts
from .evaluate_phase1c import atomic_savez, atomic_write_json
from .phase2a_env import log

TOP = 3
NAMES = ("raw_name_ratio_s1", "raw_name_eq_s1", "raw_name_ratio_sib_max", "raw_name_eq_sib", "addr_raw_ratio_s1",
         "street_sim_s1_exp", "street_sim_sib_max", "street_sim_sib_mean", "first_num_agree_sib")
ABBREVIATIONS = {"r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "blvd": "boulevard", "pl": "place",
                 "ch": "chemin", "imp": "impasse", "rte": "route", "fbg": "faubourg", "rd": "road", "ln": "lane",
                 "dr": "drive", "nr": "near", "opp": "opposite", "hwy": "highway", "sq": "square", "ct": "court",
                 "pkwy": "parkway", "cir": "circle", "mg": "marg"}
DROP = {"n", "no", "bis", "ter", "numero"}
NUMBER = re.compile(r"\d+")
WORD = re.compile(r"[^\W\d_]+")


def street_text(address: str) -> str:
    text = unicodedata.normalize("NFKD", address or "").lower()
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    words = [ABBREVIATIONS.get(w, w) for w in WORD.findall(text)]
    return " ".join(w for w in words if w not in DROP)


def first_number(address: str) -> int:
    m = NUMBER.search(address or "")
    return int(m.group().lstrip("0") or "0") if m and len(m.group()) <= 12 else -1


def siblings(codes: np.ndarray, scores: np.ndarray, top: int = TOP) -> np.ndarray:
    """(n_rows, top) row indices of each row's S1's best other rows by score (-1 padded). Rows grouped by ``codes``."""
    n = len(codes)
    filled = np.where(np.isfinite(scores), scores, -np.inf)
    order = np.lexsort((np.arange(n), -filled, codes))
    grouped = codes[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]]) if n else np.zeros(0, np.int64)
    rank = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    group_id = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
    best = np.full((len(starts), top + 1), -1, np.int64)
    keep = rank <= top
    best[group_id[keep], rank[keep]] = order[keep]
    row_group = np.empty(n, np.int64)
    row_group[order] = group_id
    cand = best[row_group]                                               # (n, top+1)
    valid = (cand >= 0) & (cand != np.arange(n)[:, None])
    moved = np.take_along_axis(cand, np.argsort(~valid, axis=1, kind="stable"), axis=1)[:, :top]
    count = valid.sum(1)
    moved[np.arange(top)[None, :] >= count[:, None]] = -1
    return moved


def read_records(paths: list[Path], keep: set) -> dict:
    out = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                if row["entity_id"] in keep:
                    out[row["entity_id"]] = (row["business_name"] or "", row["business_address"] or "")
    return out


def split_features(t_name, t_addr, s_name, s_addr, sib: np.ndarray) -> np.ndarray:
    """Arrays of per-row raw strings (object arrays) and the sibling index matrix -> feature matrix."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist

    n = len(t_name)
    X = np.full((n, len(NAMES)), np.nan, np.float32)
    t_street = np.asarray([street_text(a) for a in t_addr], dtype=object)
    s_street = np.asarray([street_text(a) for a in s_addr], dtype=object)
    t_num = np.fromiter((first_number(a) for a in t_addr), np.int64, n)
    X[:, 0] = cpdist(list(t_name), list(s_name), scorer=fuzz.ratio, dtype=np.float32, workers=-1)
    X[:, 1] = (t_name == s_name).astype(np.float32)
    t_has, s_has = np.asarray([bool(a) for a in t_addr]), np.asarray([bool(a) for a in s_addr])
    both = t_has & s_has
    X[:, 4] = np.where(both, cpdist(list(t_addr), list(s_addr), scorer=fuzz.ratio, dtype=np.float32, workers=-1), np.nan)
    X[:, 5] = np.where(both, cpdist(list(t_street), list(s_street), scorer=fuzz.token_sort_ratio, dtype=np.float32, workers=-1), np.nan)
    name_sims, name_eq, street_sims, num_agree = [], [], [], []
    for j in range(sib.shape[1]):
        idx = sib[:, j]
        ok = idx >= 0
        safe = np.where(ok, idx, 0)
        sn, sa, ss, snum = t_name[safe], t_addr[safe], t_street[safe], t_num[safe]
        ns = cpdist(list(t_name), list(sn), scorer=fuzz.ratio, dtype=np.float32, workers=-1)
        name_sims.append(np.where(ok, ns, np.nan))
        name_eq.append(np.where(ok, (t_name == sn).astype(np.float32), np.nan))
        sib_has = ok & np.asarray([bool(a) for a in sa]) & t_has
        st = cpdist(list(t_street), list(ss), scorer=fuzz.token_set_ratio, dtype=np.float32, workers=-1)
        street_sims.append(np.where(sib_has, st, np.nan))
        num_ok = ok & (t_num >= 0) & (snum >= 0)
        num_agree.append(np.where(num_ok, (t_num == snum).astype(np.float32), np.nan))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X[:, 2] = np.nanmax(np.column_stack(name_sims), axis=1)
        X[:, 3] = np.nansum(np.column_stack(name_eq), axis=1)
        X[:, 6] = np.nanmax(np.column_stack(street_sims), axis=1)
        X[:, 7] = np.nanmean(np.column_stack(street_sims), axis=1)
        X[:, 8] = np.nanmean(np.column_stack(num_agree), axis=1)
    X[:, 4:6][~both] = np.nan
    return X


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
        s_enc = table.column("s1_id").combine_chunks().dictionary_encode()
        t_enc = table.column("t_id").combine_chunks().dictionary_encode()
        s_rec = read_records([directory / f"{kind}_source1.tsv"], set(s_enc.dictionary.to_pylist()))
        t_rec = read_records([directory / f"{kind}_source{k}.tsv" for k in (2, 3)], set(t_enc.dictionary.to_pylist()))
        s_vocab, t_vocab = s_enc.dictionary.to_pylist(), t_enc.dictionary.to_pylist()
        s_codes, t_codes = s_enc.indices.to_numpy(), t_enc.indices.to_numpy()
        s_name_v = np.asarray([s_rec[x][0] for x in s_vocab], dtype=object); s_addr_v = np.asarray([s_rec[x][1] for x in s_vocab], dtype=object)
        t_name_v = np.asarray([t_rec[x][0] for x in t_vocab], dtype=object); t_addr_v = np.asarray([t_rec[x][1] for x in t_vocab], dtype=object)
        scores = np.load(args.scores_dir / f"{split}.npy")
        if len(scores) != len(s_codes):
            raise ValueError(f"{split}: {len(scores):,} scores for {len(s_codes):,} rows")
        sib = siblings(s_codes.astype(np.int64), scores)
        X = split_features(t_name_v[t_codes], t_addr_v[t_codes], s_name_v[s_codes], s_addr_v[s_codes], sib)
        atomic_savez(args.out / f"{split}.npz", {"X": X, "names": np.asarray(NAMES)})
        summary[split] = {"rows": int(len(X)), "means": np.nanmean(X, axis=0, dtype=np.float64).round(3).tolist(),
                          "nan_share": np.isnan(X).mean(0).round(3).tolist()}
        log(f"sib2_context {split}: {summary[split]}")
    atomic_write_json(args.out / "sib2_summary.json", summary)


if __name__ == "__main__":
    main()
