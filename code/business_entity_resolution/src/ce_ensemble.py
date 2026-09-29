"""Average the logits of several cross-encoder runs scored on the same pairs root (rows align by construction).

Writes ``<out>/scores/<split>.npy`` (mean logit; NaN where any run did not score the row) and ``<out>/test_k.json``
(the smallest scored K across runs), ready for ``ce_policy``.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_phase1c import atomic_write_json
from .phase2a_env import log


def average(score_dirs: list[Path], out: Path, splits: list[str]) -> dict:
    (out / "scores").mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in splits:
        arrays = [np.load(d / f"{split}.npy") for d in score_dirs]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError(f"{split}: runs have different row counts {[len(a) for a in arrays]}")
        mean = np.mean(np.stack(arrays), axis=0).astype(np.float32)      # NaN propagates: unscored anywhere -> NaN
        np.save(out / "scores" / f"{split}.npy", mean)
        summary[split] = {"rows": int(len(mean)), "scored": int(np.isfinite(mean).sum())}
    ks = [json.loads((d.parent / "test_k.json").read_text())["k"] for d in score_dirs if (d.parent / "test_k.json").exists()]
    if ks:
        atomic_write_json(out / "test_k.json", {"k": int(min(ks)), "runs": ks})
    atomic_write_json(out / "ensemble.json", {"runs": [str(d) for d in score_dirs], "splits": summary})
    log(f"ensemble of {len(score_dirs)} runs: {summary}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dirs", type=Path, nargs="+", required=True, help="<ce_out>/scores of each run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "holdout", "test"])
    args = parser.parse_args(argv)
    average(args.scores_dirs, args.out, args.splits)


if __name__ == "__main__":
    main()
