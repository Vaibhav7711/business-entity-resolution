"""Gate for downloaded cross-encoder scores: each <split>.npy must load, be finite, and have exactly the row count of
the pairs it scores (from a dense_merge_new.json / cascade_summary.json style file: {split: {"rows": n}}), or of the
pairs root itself. Exits 1 on any mismatch (a failed download can leave an HTTP error body saved as .npy).

    python cloud/check_scores.py --scores DIR --rows-json FILE --splits validation holdout test
    python cloud/check_scores.py --scores DIR --pairs-root ROOT --splits validation holdout
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def expected_rows(args, split: str) -> int:
    if args.rows_json:
        return int(json.loads(Path(args.rows_json).read_text())[split]["rows"])
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(p).metadata.num_rows for p in sorted((Path(args.pairs_root) / split).glob("part-*.parquet")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--rows-json")
    parser.add_argument("--pairs-root")
    parser.add_argument("--splits", nargs="+", required=True)
    args = parser.parse_args()
    if not (args.rows_json or args.pairs_root):
        parser.error("give --rows-json or --pairs-root")
    ok = True
    for split in args.splits:
        path = Path(args.scores) / f"{split}.npy"
        try:
            x = np.load(path)
            want = expected_rows(args, split)
            good = x.ndim == 1 and len(x) == want and bool(np.isfinite(x).all())
            print(f"{'PASS' if good else 'FAIL'} {path}: {len(x):,} rows (want {want:,}), "
                  f"{int((~np.isfinite(x)).sum()):,} non-finite")
        except Exception as error:  # noqa: BLE001
            good = False
            print(f"FAIL {path}: {type(error).__name__}: {error}")
        ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
