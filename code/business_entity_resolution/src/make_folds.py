"""Create deterministic Source-1-level cross-validation folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def stable_fold(entity_id: str, folds: int) -> int:
    digest = hashlib.blake2b(entity_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % folds


def match_bucket(value: str) -> str:
    if not value:
        return "0"
    count = value.count(",") + 1
    return str(count) if count <= 5 else "6+"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--source1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--reused-target-count",
        type=int,
        required=True,
        help="Value from Phase 0 audit. Must be zero for independent S1 hashing.",
    )
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("At least two folds are required")
    if args.reused_target_count != 0:
        raise RuntimeError(
            "Shared S2/S3 targets connect multiple S1 entities. Build connected "
            "components before assigning folds instead of using this script."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fold_counts: Counter[int] = Counter()
    fold_match_buckets: dict[int, Counter[str]] = defaultdict(Counter)

    with args.ground_truth.open("r", encoding="utf-8", newline="") as src, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "fold"])
        for row in reader:
            entity_id = row["source1_entity_id"].strip()
            fold = stable_fold(entity_id, args.folds)
            writer.writerow([entity_id, fold])
            fold_counts[fold] += 1
            fold_match_buckets[fold][match_bucket(row["matched_entity_ids"].strip())] += 1

    country_by_fold: dict[int, Counter[str]] = defaultdict(Counter)
    with args.source1.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        for row in reader:
            fold = stable_fold(row["entity_id"].strip(), args.folds)
            country_by_fold[fold][row["country"].strip() or "<EMPTY>"] += 1

    summary = {
        "folds": args.folds,
        "assignment": "blake2b(source1_entity_id) modulo folds",
        "leakage_guard": "reused_target_count verified as zero",
        "fold_counts": {str(k): fold_counts[k] for k in sorted(fold_counts)},
        "match_count_buckets": {
            str(k): dict(sorted(fold_match_buckets[k].items()))
            for k in sorted(fold_match_buckets)
        },
        "countries": {
            str(k): dict(sorted(country_by_fold[k].items()))
            for k in sorted(country_by_fold)
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

