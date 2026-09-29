"""Command-line scorer for challenge-format TSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from metrics import macro_fbeta, parse_id_list


def load_sets(path: Path, id_column: str, set_column: str) -> dict[str, set[str]]:
    rows: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = [id_column, set_column]
        if reader.fieldnames != expected:
            raise ValueError(
                f"{path}: expected columns {expected}, found {reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, start=2):
            entity_id = row[id_column].strip()
            if not entity_id:
                raise ValueError(f"{path}:{line_number}: empty {id_column}")
            if entity_id in rows:
                raise ValueError(
                    f"{path}:{line_number}: duplicate entity row {entity_id}"
                )
            rows[entity_id] = parse_id_list(row[set_column])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    args = parser.parse_args()

    truth = load_sets(
        args.truth, id_column="source1_entity_id", set_column="matched_entity_ids"
    )
    prediction = load_sets(
        args.prediction,
        id_column="source1_entity_id",
        set_column="matched_entity_ids",
    )
    score = macro_fbeta(truth, prediction, beta=0.5)
    print(f"macro_f0.5\t{score:.12f}")
    print(f"entities\t{len(truth)}")


if __name__ == "__main__":
    main()

