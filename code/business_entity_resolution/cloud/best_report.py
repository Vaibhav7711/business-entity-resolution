"""Print the name of the stack directory with the best validation out-of-fold macro F0.5 among those given, if it
beats --beat; otherwise print "none". A missing or unreadable stack_report.json never wins (fails closed).
The per-directory scores go to stderr.

    python cloud/best_report.py ~/data/stack_a ~/data/stack_b --beat 0.98080
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--beat", type=float, default=0.0)
    args = parser.parse_args()
    oof = {}
    for directory in map(Path, args.dirs):
        try:
            report = json.loads((directory / "stack_report.json").read_text())
            oof[directory.name] = float(report["validation_oof_macro_f05"])
            print(f"{directory.name}: validation OOF {oof[directory.name]:.5f} "
                  f"holdout {report['holdout']['macro_f05']:.5f}", file=sys.stderr)
        except Exception as error:  # noqa: BLE001
            print(f"{directory.name}: no usable report ({type(error).__name__})", file=sys.stderr)
    best = max(oof, key=oof.get) if oof else None
    print(best if best is not None and oof[best] > args.beat else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
