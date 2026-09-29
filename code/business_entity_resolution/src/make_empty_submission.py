"""Create a format-valid empty prediction for validator smoke testing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-source1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    with args.test_source1.open("r", encoding="utf-8", newline="") as source, matching_path.open(
        "w", encoding="utf-8", newline=""
    ) as matching, candidate_path.open("w", encoding="utf-8", newline="") as candidate:
        reader = csv.DictReader(source, delimiter="\t")
        matching_writer = csv.writer(matching, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for row in reader:
            entity_id = row["entity_id"]
            matching_writer.writerow([entity_id, ""])
            candidate_writer.writerow([entity_id, ""])

    print(matching_path)
    print(candidate_path)


if __name__ == "__main__":
    main()
