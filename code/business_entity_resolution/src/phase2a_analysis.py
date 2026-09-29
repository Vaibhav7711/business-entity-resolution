"""Phase 2A error-structure analysis on the local benchmark pairs (read-only inputs).

    python3 -m src.phase2a_analysis --config ../../configs/phase2a_benchmark.json --part structure
    python3 -m src.phase2a_analysis --config ../../configs/phase2a_benchmark.json --part diagnostic

``structure``: target ownership (each S2/S3 record belongs to at most one S1),
hard-negative strata, and a noise-pattern audit of gold pairs; no model.
``diagnostic``: a reduced LightGBM (same features/params, fewer train S1) whose
per-pair errors are categorised and costed by entity-level F0.5 counterfactuals.
Outputs go to artifacts/phase2a_analysis/ only.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import json
import re
import unicodedata
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from .blocking import encode_id
from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json
from .phase2a_env import ResourceGuard, ResourceStop, log
from .phase2a_eval import apply_policy, entity_fbeta, sweep
from .phase2a_pairs import FEATURES, Layout, TargetStore, split_of

F = {name: i for i, name in enumerate(FEATURES)}
DIGITS = re.compile(r"\d+")
EVAL_SPLITS = ("validation", "holdout")


def eval_candidate_ids(layout: Layout, config: dict) -> np.ndarray:
    """Sorted unique candidate IDs over all evaluation chunks (bounds the owner map)."""
    union = np.zeros(0, dtype=np.uint32)
    for chunk, _ in eval_chunks(layout, config):
        with np.load(layout.pair_chunk(chunk), allow_pickle=False) as data:
            union = np.union1d(union, np.unique(data["cand"]))
    return union


def owner_map(train_dir: Path, wanted: np.ndarray | None = None,
              batch: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Owning numeric S1 ID (from training truth) for each wanted target, streamed in batches."""
    kept_t, kept_o = [], []
    targets, owners = array("I"), array("i")

    def flush():
        t = np.frombuffer(targets, dtype=np.uint32).copy()
        o = np.frombuffer(owners, dtype=np.int32).copy()
        keep = np.ones(len(t), bool) if wanted is None else np.isin(t, wanted, assume_unique=False)
        kept_t.append(t[keep]); kept_o.append(o[keep])
        del targets[:], owners[:]

    with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)
        for s1, matched in reader:
            if matched:
                owner = int(s1.split("-", 1)[1])
                for value in matched.split(","):
                    targets.append(encode_id(value))
                    owners.append(owner)
                if len(targets) >= batch:
                    flush()
    flush()
    t, o = np.concatenate(kept_t), np.concatenate(kept_o)
    order = np.argsort(t, kind="stable")
    return t[order], o[order]


class MmapTargetStore(TargetStore):
    """TargetStore whose index arrays are memory-mapped .npy files (built once from the npz)."""

    def __init__(self, texts: Path, cache: Path):
        cache.mkdir(parents=True, exist_ok=True)
        names = ("sorted_ids", "order", "name_offsets", "address_offsets", "non_ascii", "address_missing", "name_missing")
        if not all((cache / f"{name}.npy").exists() for name in names):
            with np.load(texts / "targets_meta.npz", allow_pickle=False) as data:
                for name in names:
                    np.save(cache / f"{name}.npy", data[name])
        self.meta = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in names}
        import os
        self.files = {"name": os.open(texts / "targets_name.bin", os.O_RDONLY),
                      "address": os.open(texts / "targets_address.bin", os.O_RDONLY)}


def lookup_owner(sorted_targets: np.ndarray, owners: np.ndarray, cand: np.ndarray) -> np.ndarray:
    where = np.minimum(np.searchsorted(sorted_targets, cand), len(sorted_targets) - 1)
    return np.where(sorted_targets[where] == cand, owners[where], -1)


def eval_chunks(layout: Layout, config: dict) -> list[tuple[int, str]]:
    result = []
    for path in sorted(layout.features.glob("chunk*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        split = split_of(meta["chunk"] * config["chunk_s1"], config["split"])
        if split in EVAL_SPLITS:
            result.append((meta["chunk"], split))
    return result


def strata(X: np.ndarray) -> dict[str, np.ndarray]:
    digits_same = np.nan_to_num(X[:, F["digits_exact"]]) == 1
    address_same = digits_same & (np.nan_to_num(X[:, F["addr_token_set"]]) >= 0.9)
    name_strong = np.nan_to_num(X[:, F["name_token_set"]]) >= 0.9
    return {"same_address": address_same, "strong_name": name_strong,
            "same_address_and_strong_name": address_same & name_strong,
            "same_address_weak_name": address_same & (np.nan_to_num(X[:, F["name_token_set"]]) < 0.6)}


def script_of(text: str) -> str:
    for char in text:
        if char.isalpha() and ord(char) > 127:
            name = unicodedata.name(char, "")
            return name.split(" ")[0] if name else "OTHER"
    return "LATIN"


def canon_digits(tokens: list[str]) -> set[str]:
    return {token.lstrip("0") or "0" for token in tokens}


def pattern_audit(s1_name: str, s1_addr: str, t_name: str, t_addr: str) -> dict:
    d1, d2 = DIGITS.findall(s1_addr), DIGITS.findall(t_addr)
    s1d, s2d = set(d1), set(d2)
    c1, c2 = canon_digits(d1), canon_digits(d2)
    only1, only2 = c1 - c2, c2 - c1
    one_edit = bool(only1 and only2) and any(Levenshtein.distance(a, b) == 1 for a in only1 for b in only2)
    return {
        "target_address_empty": not t_addr,
        "digits_equal": bool(d1 or d2) and s1d == s2d,
        "digits_equal_after_leading_zero_strip": bool(d1 or d2) and s1d != s2d and c1 == c2,
        "digits_target_subset": bool(c2) and c2 < c1,
        "digits_one_edit_typo": one_edit,
        "digits_other_mismatch": bool(d1 or d2) and c1 != c2 and not (bool(c2) and c2 < c1) and not one_edit,
        "target_name_script": script_of(t_name),
        "name_token_set": fuzz.token_set_ratio(s1_name, t_name),
    }


def structure(config: dict, layout: Layout, guard: ResourceGuard) -> dict:
    wanted = eval_candidate_ids(layout, config)
    guard.check("candidate union")
    sorted_targets, owners = owner_map(ROOT / config["inputs"]["train_dir"], wanted)
    del wanted
    guard.check("owner map")
    with np.load(layout.texts / "s1.npz", allow_pickle=False) as data:
        s1 = {name: data[name].tolist() for name in ("entity_id", "name", "address")}
    s1_num = np.asarray([int(value.split("-", 1)[1]) for value in s1["entity_id"]], dtype=np.int64)
    store = MmapTargetStore(layout.texts, ROOT / "artifacts/phase2a_analysis/cache")
    table: Counter = Counter()
    per_split_s1 = Counter()
    singleton_top = Counter()
    token_pairs, s1_only, tgt_only = Counter(), Counter(), Counter()
    patterns: Counter = Counter()
    scripts: Counter = Counter()
    name_bins: Counter = Counter()
    positives_seen = 0
    for chunk, split in eval_chunks(layout, config):
        guard.check(f"structure chunk {chunk}")
        with np.load(layout.pair_chunk(chunk), allow_pickle=False) as data:
            pairs = {name: data[name] for name in ("cand", "label", "s1_pos", "rrf_rank", "s1_truth_len", "s1_positions")}
        with np.load(layout.feature_chunk(chunk), allow_pickle=False) as data:
            X = data["X"]
            rows = data["rows"]
        if not np.array_equal(rows, np.arange(len(pairs["cand"]))):
            raise AssertionError("evaluation feature chunks must cover every pair")
        owner = lookup_owner(sorted_targets, owners, pairs["cand"])
        this_s1 = s1_num[pairs["s1_pos"]]
        label = pairs["label"]
        kind = np.where(label, "positive", np.where(owner < 0, "neg_unowned", "neg_owned_by_other_s1"))
        if np.any(label & (owner != this_s1)):
            raise AssertionError("a positive pair is not owned by its S1")
        per_split_s1[split] += len(pairs["s1_positions"])
        for stratum, mask in {"all": np.ones(len(label), bool), **strata(X)}.items():
            for k in ("positive", "neg_unowned", "neg_owned_by_other_s1"):
                table[split, stratum, k] += int(np.sum(mask & (kind == k)))
        # Singletons: what do their strongest look-alikes belong to?
        singleton_rows = pairs["s1_truth_len"][pairs["s1_pos"] - pairs["s1_positions"][0]] == 0
        hard = strata(X)["same_address_and_strong_name"]
        for k in ("neg_unowned", "neg_owned_by_other_s1"):
            singleton_top[split, k] += int(np.sum(singleton_rows & hard & (kind == k)))
        singleton_top[split, "singletons_with_same_address_strong_name_candidate"] += int(len(np.unique(
            pairs["s1_pos"][singleton_rows & hard])))
        # Noise-pattern audit on gold pairs.
        for i in np.flatnonzero(label):
            p = int(pairs["s1_pos"][i])
            row = int(store.rows(pairs["cand"][i:i + 1])[0])
            t_name, t_addr = store.text(row, "name"), store.text(row, "address")
            audit = pattern_audit(s1["name"][p], s1["address"][p], t_name, t_addr)
            positives_seen += 1
            for key, value in audit.items():
                if isinstance(value, bool) and value:
                    patterns[key] += 1
            scripts[audit["target_name_script"]] += 1
            name_bins["<50" if audit["name_token_set"] < 50 else "50-89" if audit["name_token_set"] < 90 else ">=90"] += 1
            a_tokens = set(s1["address"][p].split())
            b_tokens = set(t_addr.split())
            only_a = {t for t in a_tokens - b_tokens if not t.isdigit()}
            only_b = {t for t in b_tokens - a_tokens if not t.isdigit()}
            s1_only.update(only_a)
            tgt_only.update(only_b)
            if 0 < len(only_a) <= 2 and 0 < len(only_b) <= 2:
                token_pairs.update((a, b) for a in only_a for b in only_b)
        log(f"structure: chunk {chunk} ({split}) done")
    rows_out = {}
    for (split, stratum, k), value in sorted(table.items()):
        rows_out.setdefault(split, {}).setdefault(stratum, {})[k] = value
    for split in rows_out:
        for stratum, counts in rows_out[split].items():
            negatives = counts.get("neg_unowned", 0) + counts.get("neg_owned_by_other_s1", 0)
            counts["share_of_negatives_owned_by_other_s1"] = (counts.get("neg_owned_by_other_s1", 0) / negatives
                                                              if negatives else None)
            counts["positive_rate"] = counts.get("positive", 0) / (negatives + counts.get("positive", 0)) \
                if negatives + counts.get("positive", 0) else None
    return {
        "evaluated_s1": dict(per_split_s1),
        "pair_ownership_by_stratum": rows_out,
        "singleton_lookalikes": {f"{k[0]}/{k[1]}": v for k, v in singleton_top.items()},
        "gold_pair_patterns": {"gold_pairs": positives_seen, **{k: {"count": v, "share": v / positives_seen}
                                                                for k, v in patterns.most_common()}},
        "gold_target_name_script": dict(scripts.most_common()),
        "gold_name_token_set_bins": dict(name_bins),
        "address_token_substitutions_top": [[a, b, n] for (a, b), n in token_pairs.most_common(60)],
        "address_tokens_only_in_s1_top": s1_only.most_common(30),
        "address_tokens_only_in_target_top": tgt_only.most_common(30),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--part", choices=("structure", "diagnostic"), required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.resolve().read_text())
    layout = Layout(ROOT / config["paths"]["output_dir"])
    out = ROOT / "artifacts/phase2a_analysis"
    resources = config["resources"]
    guard = ResourceGuard(resources["min_available_memory_gib"], resources["max_swap_growth_gib"],
                          f"cd code/business_entity_resolution && python3 -m src.phase2a_analysis --config {args.config} --part {args.part}")
    try:
        if args.part == "structure":
            result = structure(config, layout, guard)
        else:
            result = diagnostic(config, layout, guard)
        result["resources"] = guard.summary()
        atomic_write_json(out / f"{args.part}.json", result)
        log(f"wrote {out / (args.part + '.json')}")
    except ResourceStop as stop:
        log(str(stop))
        raise SystemExit(3)


def diagnostic(config: dict, layout: Layout, guard: ResourceGuard) -> dict:  # defined below
    raise NotImplementedError


if __name__ == "__main__":
    main()
