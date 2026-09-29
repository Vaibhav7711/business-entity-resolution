"""Low-disk streaming Phase 0 audit for the entity-resolution dataset.

Exact ID/label checks use compact bitsets. Bloom filters provide low-disk
duplicate diagnostics, and positive-pair agreement is measured on a stable S1
sample. The supplied TSV files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path


SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_HEADER = ["source1_entity_id", "matched_entity_ids"]
ID_SPACE = 1_000_000_000
BITSET_BYTES = (ID_SPACE + 7) // 8
NON_WORD = re.compile(r"[^\w]+", flags=re.UNICODE)


class BloomFilter:
    """Compact probabilistic membership filter using double hashing."""

    def __init__(self, bit_power: int = 28, hashes: int = 4) -> None:
        self.bit_count = 1 << bit_power
        self.mask = self.bit_count - 1
        self.bits = bytearray(self.bit_count >> 3)
        self.hashes = hashes
        self.insertions = 0

    @staticmethod
    def digest(value: str) -> bytes:
        return hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest()

    def _positions(self, digest: bytes):
        left = int.from_bytes(digest[:8], "little")
        right = int.from_bytes(digest[8:], "little") | 1
        for index in range(self.hashes):
            yield (left + index * right) & self.mask

    def contains_digest(self, digest: bytes) -> bool:
        return all(
            self.bits[position >> 3] & (1 << (position & 7))
            for position in self._positions(digest)
        )

    def add_digest(self, digest: bytes) -> None:
        for position in self._positions(digest):
            self.bits[position >> 3] |= 1 << (position & 7)
        self.insertions += 1

    def check_and_add(self, digest: bytes) -> bool:
        present = self.contains_digest(digest)
        self.add_digest(digest)
        return present

    def estimated_false_positive_rate(self) -> float:
        exponent = -self.hashes * self.insertions / self.bit_count
        return (1.0 - math.exp(exponent)) ** self.hashes


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    value = NON_WORD.sub(" ", value).replace("_", " ")
    return " ".join(value.split())


def field_signature(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little"
    )


def record_signature(name: str, address: str, country: str) -> bytes:
    return BloomFilter.digest("\x1f".join((name, address, country)))


def compact_pair_signature(name: str, address: str, country: str) -> tuple[int, ...]:
    norm_name = normalize_text(name)
    norm_address = normalize_text(address)
    norm_country = normalize_text(country)
    return (
        field_signature(name),
        field_signature(norm_name),
        field_signature(address),
        field_signature(norm_address),
        field_signature(norm_country),
        int(not name.strip()),
        int(not address.strip()),
        int(not country.strip()),
    )


def parse_numeric_id(entity_id: str, expected_prefix: str) -> int | None:
    prefix = expected_prefix + "-"
    if not entity_id.startswith(prefix):
        return None
    suffix = entity_id[len(prefix) :]
    if not suffix.isdigit():
        return None
    number = int(suffix)
    return number if 0 <= number < ID_SPACE else None


def bit_is_set(bits: bytearray, number: int) -> bool:
    return bool(bits[number >> 3] & (1 << (number & 7)))


def set_bit(bits: bytearray, number: int) -> None:
    bits[number >> 3] |= 1 << (number & 7)


def quantile_from_hist(hist: Counter[int], q: float) -> int | None:
    total = sum(hist.values())
    if not total:
        return None
    target = max(1, int((total - 1) * q) + 1)
    cumulative = 0
    for value in sorted(hist):
        cumulative += hist[value]
        if cumulative >= target:
            return value
    return max(hist)


def audit_source_file(
    path: Path,
    expected_prefix: str,
    other_split_seen: bytearray | None,
    train_raw_bloom: BloomFilter | None,
    train_norm_bloom: BloomFilter | None,
    sample_modulo: int,
    collect_s1_sample: bool,
) -> tuple[dict, bytearray, BloomFilter, BloomFilter, dict[int, tuple[int, ...]]]:
    local_seen = bytearray(BITSET_BYTES)
    local_raw_bloom = BloomFilter()
    local_norm_bloom = BloomFilter()
    sampled_signatures: dict[int, tuple[int, ...]] = {}
    stats: dict = {
        "path": str(path),
        "rows": 0,
        "bad_column_count": 0,
        "invalid_ids": 0,
        "duplicate_ids_within_file": 0,
        "id_overlap_with_other_split": 0,
        "probable_exact_duplicate_rows": 0,
        "probable_normalized_collision_rows": 0,
        "probable_cross_split_exact_rows": 0,
        "probable_cross_split_normalized_rows": 0,
        "missing": Counter(),
        "countries": Counter(),
        "non_ascii_name_rows": 0,
        "non_ascii_address_rows": 0,
    }
    name_lengths: Counter[int] = Counter()
    address_lengths: Counter[int] = Counter()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != SOURCE_HEADER:
            raise ValueError(f"{path}: expected header {SOURCE_HEADER}, found {header}")
        for row in reader:
            stats["rows"] += 1
            if len(row) != 4:
                stats["bad_column_count"] += 1
                continue
            entity_id, name, address, country = row
            number = parse_numeric_id(entity_id, expected_prefix)
            if number is None:
                stats["invalid_ids"] += 1
            else:
                if bit_is_set(local_seen, number):
                    stats["duplicate_ids_within_file"] += 1
                else:
                    set_bit(local_seen, number)
                if other_split_seen is not None and bit_is_set(other_split_seen, number):
                    stats["id_overlap_with_other_split"] += 1

            name_missing = not name.strip()
            address_missing = not address.strip()
            country_missing = not country.strip()
            stats["missing"]["business_name"] += int(name_missing)
            stats["missing"]["business_address"] += int(address_missing)
            stats["missing"]["country"] += int(country_missing)
            stats["countries"][country.strip() or "<EMPTY>"] += 1
            stats["non_ascii_name_rows"] += int(not name.isascii())
            stats["non_ascii_address_rows"] += int(not address.isascii())
            name_lengths[len(name)] += 1
            address_lengths[len(address)] += 1

            norm_name = normalize_text(name)
            norm_address = normalize_text(address)
            norm_country = normalize_text(country)
            raw_digest = record_signature(name, address, country)
            norm_digest = record_signature(norm_name, norm_address, norm_country)
            stats["probable_exact_duplicate_rows"] += int(
                local_raw_bloom.check_and_add(raw_digest)
            )
            stats["probable_normalized_collision_rows"] += int(
                local_norm_bloom.check_and_add(norm_digest)
            )
            if train_raw_bloom is not None and train_norm_bloom is not None:
                stats["probable_cross_split_exact_rows"] += int(
                    train_raw_bloom.contains_digest(raw_digest)
                )
                stats["probable_cross_split_normalized_rows"] += int(
                    train_norm_bloom.contains_digest(norm_digest)
                )

            if (
                collect_s1_sample
                and number is not None
                and field_signature(entity_id) % sample_modulo == 0
            ):
                sampled_signatures[number] = compact_pair_signature(name, address, country)

    def length_summary(hist: Counter[int]) -> dict:
        total = sum(hist.values())
        weighted = sum(length * count for length, count in hist.items())
        return {
            "mean": weighted / total if total else None,
            "p50": quantile_from_hist(hist, 0.50),
            "p95": quantile_from_hist(hist, 0.95),
            "max": max(hist) if hist else None,
        }

    stats["missing"] = dict(stats["missing"])
    stats["countries"] = dict(stats["countries"])
    stats["name_length"] = length_summary(name_lengths)
    stats["address_length"] = length_summary(address_lengths)
    stats["bloom_false_positive_rate"] = {
        "raw": local_raw_bloom.estimated_false_positive_rate(),
        "normalized": local_norm_bloom.estimated_false_positive_rate(),
    }
    return stats, local_seen, local_raw_bloom, local_norm_bloom, sampled_signatures


def audit_ground_truth(
    path: Path,
    train_seen: dict[str, bytearray],
    sampled_s1: dict[int, tuple[int, ...]],
) -> tuple[dict, dict[str, dict[int, tuple[int, ...]]]]:
    stats: dict = {
        "rows": 0,
        "duplicate_source1_rows": 0,
        "source1_not_in_train_source1": 0,
        "invalid_source1_ids": 0,
        "duplicate_ids_inside_match_list": 0,
        "invalid_target_ids": 0,
        "target_ids_missing_from_train_source": 0,
        "targets_reused_across_source1_entities": 0,
        "match_count_distribution": Counter(),
        "links_by_source": Counter(),
        "sampled_source1_entities": 0,
        "sampled_positive_links": 0,
    }
    gt_seen = bytearray(BITSET_BYTES)
    target_seen = {"S2": bytearray(BITSET_BYTES), "S3": bytearray(BITSET_BYTES)}
    sampled_targets: dict[str, dict[int, tuple[int, ...]]] = {"S2": {}, "S3": {}}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != GROUND_TRUTH_HEADER:
            raise ValueError(f"{path}: expected header {GROUND_TRUTH_HEADER}, found {header}")
        for row in reader:
            stats["rows"] += 1
            if len(row) == 1:
                row.append("")
            if len(row) != 2:
                raise ValueError(f"{path}: malformed row {stats['rows'] + 1}")
            source1_id, values = row
            source1_number = parse_numeric_id(source1_id, "S1")
            sampled_signature = None
            if source1_number is None:
                stats["invalid_source1_ids"] += 1
            else:
                if bit_is_set(gt_seen, source1_number):
                    stats["duplicate_source1_rows"] += 1
                else:
                    set_bit(gt_seen, source1_number)
                if not bit_is_set(train_seen["S1"], source1_number):
                    stats["source1_not_in_train_source1"] += 1
                sampled_signature = sampled_s1.get(source1_number)
                stats["sampled_source1_entities"] += int(sampled_signature is not None)

            targets = [] if not values.strip() else [v.strip() for v in values.split(",")]
            stats["match_count_distribution"][len(targets)] += 1
            if len(targets) != len(set(targets)):
                stats["duplicate_ids_inside_match_list"] += 1

            seen_in_row: set[str] = set()
            for target_id in targets:
                if target_id in seen_in_row:
                    continue
                seen_in_row.add(target_id)
                if target_id.startswith("S2-"):
                    source = "S2"
                elif target_id.startswith("S3-"):
                    source = "S3"
                else:
                    stats["invalid_target_ids"] += 1
                    continue
                number = parse_numeric_id(target_id, source)
                if number is None:
                    stats["invalid_target_ids"] += 1
                    continue
                if not bit_is_set(train_seen[source], number):
                    stats["target_ids_missing_from_train_source"] += 1
                if bit_is_set(target_seen[source], number):
                    stats["targets_reused_across_source1_entities"] += 1
                else:
                    set_bit(target_seen[source], number)
                stats["links_by_source"][source] += 1
                if sampled_signature is not None:
                    sampled_targets[source][number] = sampled_signature
                    stats["sampled_positive_links"] += 1

    stats["match_count_distribution"] = {
        str(k): v for k, v in sorted(stats["match_count_distribution"].items())
    }
    stats["links_by_source"] = dict(stats["links_by_source"])
    return stats, sampled_targets


def audit_sampled_positive_agreement(
    path: Path, expected_prefix: str, sampled_targets: dict[int, tuple[int, ...]]
) -> dict:
    counts: Counter[str] = Counter()
    found: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != SOURCE_HEADER:
            raise ValueError(f"{path}: unexpected header")
        for row in reader:
            if len(row) != 4:
                continue
            entity_id, name, address, country = row
            number = parse_numeric_id(entity_id, expected_prefix)
            if number is None or number not in sampled_targets:
                continue
            found.add(number)
            s1 = sampled_targets[number]
            target = compact_pair_signature(name, address, country)
            counts["pairs"] += 1
            counts["name_raw_equal"] += int(s1[0] == target[0])
            counts["name_normalized_equal"] += int(s1[1] == target[1])
            counts["address_raw_equal"] += int(s1[2] == target[2])
            counts["address_normalized_equal"] += int(s1[3] == target[3])
            counts["country_normalized_equal"] += int(s1[4] == target[4])
            counts["both_name_address_normalized_equal"] += int(
                s1[1] == target[1] and s1[3] == target[3]
            )
            counts["s1_name_missing"] += s1[5]
            counts["target_name_missing"] += target[5]
            counts["s1_address_missing"] += s1[6]
            counts["target_address_missing"] += target[6]
    counts["sampled_target_ids"] = len(sampled_targets)
    counts["sampled_target_ids_not_found"] = len(sampled_targets) - len(found)
    return dict(counts)


def percentage(numerator: int, denominator: int) -> str:
    return "n.a." if not denominator else f"{100 * numerator / denominator:.3f}%"


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Phase 0 data audit", "", "## Scale", "", "| Split | Source | Rows |", "|---|---:|---:|"]
    for split in ("train", "test"):
        for source in ("S1", "S2", "S3"):
            lines.append(f"| {split} | {source} | {report['sources'][split][source]['rows']:,} |")

    gt = report["ground_truth"]
    singleton_count = int(gt["match_count_distribution"].get("0", 0))
    total_links = sum(gt["links_by_source"].values())
    lines += [
        "", "## Ground truth", "",
        f"- Source 1 entities: {gt['rows']:,}",
        f"- Singleton entities: {singleton_count:,} ({percentage(singleton_count, gt['rows'])})",
        f"- Positive links: {total_links:,}",
        f"- Mean links per Source 1 entity: {total_links / gt['rows']:.3f}",
        f"- Maximum links for one Source 1 entity: {max(map(int, gt['match_count_distribution']))}",
        f"- S2 links: {gt['links_by_source'].get('S2', 0):,}",
        f"- S3 links: {gt['links_by_source'].get('S3', 0):,}",
        "", "### Match-count distribution", "",
        "| True matches | S1 entities | Share |", "|---:|---:|---:|",
    ]
    for count, entities in gt["match_count_distribution"].items():
        lines.append(f"| {count} | {entities:,} | {percentage(entities, gt['rows'])} |")

    lines += ["", "## Countries and missingness", ""]
    for split in ("train", "test"):
        lines += [
            f"### {split.title()}", "",
            "| Source | Countries | Missing name | Missing address | Non-ASCII name |",
            "|---|---|---:|---:|---:|",
        ]
        for source in ("S1", "S2", "S3"):
            stats = report["sources"][split][source]
            countries = ", ".join(f"{k}: {v:,}" for k, v in sorted(stats["countries"].items()))
            rows = stats["rows"]
            lines.append(
                f"| {source} | {countries} | {percentage(stats['missing']['business_name'], rows)} | "
                f"{percentage(stats['missing']['business_address'], rows)} | "
                f"{percentage(stats['non_ascii_name_rows'], rows)} |"
            )
        lines.append("")

    lines += ["## Integrity checks", ""]
    for key, value in report["integrity"].items():
        lines.append(f"- {key.replace('_', ' ').capitalize()}: {value:,}")

    lines += [
        "", "## Probabilistic duplicate and train/test overlap diagnostics", "",
        "These are Bloom-filter counts and may contain a small number of false positives.", "",
        "| Split | Source | Probable exact duplicates | Probable normalized collisions | Probable cross-split exact rows | Probable cross-split normalized rows |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "test"):
        for source in ("S1", "S2", "S3"):
            stats = report["sources"][split][source]
            lines.append(
                f"| {split} | {source} | {stats['probable_exact_duplicate_rows']:,} | "
                f"{stats['probable_normalized_collision_rows']:,} | "
                f"{stats['probable_cross_split_exact_rows']:,} | "
                f"{stats['probable_cross_split_normalized_rows']:,} |"
            )

    lines += [
        "", "## Positive-pair exact agreement", "",
        f"Deterministic sample: approximately 1/{report['sample_modulo']} of Source 1 IDs.", "",
        "| Source | Signal | Count | Rate |", "|---|---|---:|---:|",
    ]
    for source in ("S2", "S3"):
        agreement = report["positive_agreement"][source]
        pairs = agreement.get("pairs", 0)
        for signal in (
            "name_raw_equal", "name_normalized_equal", "address_raw_equal",
            "address_normalized_equal", "both_name_address_normalized_equal",
            "country_normalized_equal",
        ):
            count = agreement.get(signal, 0)
            lines.append(f"| {source} | {signal.replace('_', ' ')} | {count:,} | {percentage(count, pairs)} |")

    lines += [
        "", "## Method notes", "",
        "- ID uniqueness, ID membership, ground-truth coverage, and target reuse checks are exact.",
        "- Record duplicate and cross-split overlap counts are probabilistic Bloom-filter diagnostics; per-file false-positive estimates are in the JSON report.",
        "- Positive agreement rates use a deterministic hash sample of Source 1 entities and exact equality of compact field signatures.",
        "- Text normalization uses Unicode NFKC, case folding, `&` to `and`, non-word removal, and whitespace collapse.",
        "- No external data, API, geocoder, pretrained model, or language resource is used.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-modulo", type=int, default=20)
    args = parser.parse_args()
    if args.sample_modulo < 1:
        raise ValueError("sample-modulo must be positive")

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "dataset_dir": str(dataset_dir),
        "sample_modulo": args.sample_modulo,
        "sources": {"train": {}, "test": {}},
    }
    train_seen: dict[str, bytearray] = {}
    sampled_s1: dict[int, tuple[int, ...]] = {}

    for source in ("S1", "S2", "S3"):
        number = source[-1]
        train_path = dataset_dir / "train" / f"train_source{number}.tsv"
        test_path = dataset_dir / "test" / f"test_source{number}.tsv"
        train_stats, train_bits, raw_bloom, norm_bloom, source_sample = audit_source_file(
            train_path, source, None, None, None, args.sample_modulo, source == "S1"
        )
        test_stats, _test_bits, _test_raw, _test_norm, _ = audit_source_file(
            test_path, source, train_bits, raw_bloom, norm_bloom, args.sample_modulo, False
        )
        train_seen[source] = train_bits
        if source == "S1":
            sampled_s1 = source_sample
        report["sources"]["train"][source] = train_stats
        report["sources"]["test"][source] = test_stats

    ground_truth, sampled_targets = audit_ground_truth(
        dataset_dir / "train" / "train_ground_truth.tsv", train_seen, sampled_s1
    )
    report["ground_truth"] = ground_truth
    report["positive_agreement"] = {}
    for source in ("S2", "S3"):
        report["positive_agreement"][source] = audit_sampled_positive_agreement(
            dataset_dir / "train" / f"train_source{source[-1]}.tsv",
            source,
            sampled_targets[source],
        )

    integrity = Counter()
    for split in ("train", "test"):
        for source in ("S1", "S2", "S3"):
            stats = report["sources"][split][source]
            integrity["bad_source_rows"] += stats["bad_column_count"]
            integrity["invalid_entity_ids"] += stats["invalid_ids"]
            integrity["duplicate_entity_ids_within_files"] += stats["duplicate_ids_within_file"]
            integrity["train_test_entity_id_overlaps"] += stats["id_overlap_with_other_split"]
    for key in (
        "duplicate_source1_rows", "source1_not_in_train_source1", "invalid_source1_ids",
        "duplicate_ids_inside_match_list", "invalid_target_ids",
        "target_ids_missing_from_train_source", "targets_reused_across_source1_entities",
    ):
        integrity[key] += ground_truth[key]
    report["integrity"] = dict(integrity)

    json_path = output_dir / "phase0_audit.json"
    markdown_path = output_dir / "phase0_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
