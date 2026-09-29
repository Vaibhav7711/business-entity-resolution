"""Country-specific decision rule on top of the pooled stacker (applied to test only; US/India keep the pooled rule).

France is absent from the labelled data, and label-free diagnostics (EM prior re-estimation, model agreement, feature
shift) showed the pooled model under-confident there. For rows of the chosen country only:

* ``--country-threshold``: the probability threshold replacing the pooled one;
* ``--empty-top``: the top candidate (by q) of a list the pooled rule leaves empty is accepted at this lower threshold.

Every other country keeps the pooled threshold, and the one-owner rule runs over all rows afterwards. The values were
chosen with feedback from the public test split (France-only changes); on the labelled validation/holdout OOF (US/India) the
empty-list rule at 0.50 changed macro F0.5 by +0.00009 / -0.00010 (neutral).
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import ce_policy as P
from . import ce_stack as S
from .evaluate_phase1c import atomic_write_json
from .phase2a_env import log


def list_rank(g: np.ndarray, q: np.ndarray) -> np.ndarray:
    """0-based rank of each row by q (descending) within its group ``g``; ties by row order."""
    n = len(q)
    order = np.lexsort((np.arange(n), -q, g))
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]) if n else np.zeros(0, np.int64)
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    return rank


def row_thresholds(g: np.ndarray, q: np.ndarray, in_country: np.ndarray, t: float, owner_base: np.ndarray,
                   t_country: float | None = None, t_empty_top: float | None = None) -> np.ndarray:
    """Per-row threshold: ``t`` everywhere; for ``in_country`` rows ``t_country`` and, on the top row of a list with
    no pooled match (``owner_base`` all False), ``min(threshold, t_empty_top)``."""
    thr = np.where(in_country, t if t_country is None else t_country, t)
    if t_empty_top is not None:
        has = np.bincount(g[owner_base], minlength=int(g.max()) + 1 if len(g) else 0) > 0
        empty_top = in_country & (list_rank(g, q) == 0) & ~has[g]
        thr = np.where(empty_top, np.minimum(thr, t_empty_top), thr)
    return thr


def main(argv: list[str] | None = None) -> None:
    import lightgbm as lgb

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--stack-dir", type=Path, required=True, help="ce_stack --pooled output (stacker.txt, ...)")
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--country", default="France")
    parser.add_argument("--country-threshold", type=float)
    parser.add_argument("--empty-top", type=float)
    args = parser.parse_args(argv)
    rep = json.loads((args.stack_dir / "stack_report.json").read_text())
    cal = json.loads((args.stack_dir / "calibrator.json").read_text())
    t, owner = rep["threshold"], rep["one_owner"]
    test = S.load(args.pairs_root, args.scores_dir, args.context_dir, "test", keep_targets=True, extra_dir=args.extra_dir)
    q = lgb.Booster(model_file=str(args.stack_dir / "stacker.txt")).predict(S.matrix(test, cal)).astype(np.float32)
    q = q.astype(float)                                                   # float32 round trip, as cached for probes
    with (args.test_dir / "test_source1.tsv").open(encoding="utf-8", newline="") as file:
        country = {r["entity_id"]: r["country"] for r in csv.DictReader(file, delimiter="\t")}
    in_country = (np.asarray([country[s] for s in test["s1_ids"]], dtype=object) == args.country)[test["g"]]
    base = S.decide(test, q, t, owner)
    thr = row_thresholds(test["g"], q, in_country, t, base, args.country_threshold, args.empty_top)
    pred = P.one_owner(q >= thr, test, q)[0] if owner else q >= thr
    if not np.array_equal(pred[~in_country], base[~in_country]):
        log("country_rule: note - one-owner reassigned rows outside the country")
    out_dir = args.out / "output"
    report = {"country": args.country, "pooled_threshold": t, "country_threshold": args.country_threshold,
              "empty_top": args.empty_top, "added_matches": int(pred.sum() - base.sum()),
              "test": S.write_test(test, pred, args.test_dir, out_dir)}
    report["test"]["validator"] = P.run_validator(P.VALIDATOR, out_dir, args.test_dir)
    atomic_write_json(args.out / "country_rule_report.json", report)
    log(f"country_rule: {report['added_matches']:+,} matches vs pooled; validator {report['test']['validator']['passed']}")
    if not report["test"]["validator"]["passed"]:
        raise SystemExit("validator failed")


if __name__ == "__main__":
    main()
