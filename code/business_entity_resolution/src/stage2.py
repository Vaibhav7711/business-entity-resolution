"""Second-stage decision model over list-level aggregates of the first-stage stacker's probabilities.

Stage 1 is the pooled ``ce_stack`` model (validation + holdout, folds by S1). Its out-of-fold probability q of every
row is summarised within the row's S1 list (rank, gap to the list's best, second best, sum, counts above 0.5 / 0.9,
mean of the others) and a second LightGBM learns from [stage-1 features, q, aggregates] with the same folds.
On test, q is the average of the 4 fold models (the kind of model that produced the out-of-fold q), so stage 2
sees the same q distribution on every split. The threshold and one-owner choice follow ``ce_stack.run_pooled``;
the report compares stage 2 with stage 1 on the same out-of-fold predictions.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import ce_policy as P
from . import ce_stack as S
from .evaluate_phase1c import atomic_write_json
from .phase2a_env import log

AGG_NAMES = ("q", "q_rank", "q_gap_max", "q_best_other", "q_sum", "q_n05", "q_n09", "q_mean_other")


def aggregates(g: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per row: q and its S1 list context (rows grouped by ``g``, any order)."""
    n = len(q)
    order = np.lexsort((-q, g))
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]) if n else np.zeros(0, np.int64)
    sizes = np.diff(np.r_[starts, n])
    group = np.repeat(np.arange(len(starts)), sizes)
    rank_sorted = np.arange(n) - np.repeat(starts, sizes)
    first = q[order][starts]
    second = np.where(sizes > 1, q[order][np.minimum(starts + 1, n - 1)], np.nan)
    qs = q[order]
    total = np.add.reduceat(qs, starts) if n else np.zeros(0)
    n05 = np.add.reduceat((qs >= 0.5).astype(float), starts) if n else np.zeros(0)
    n09 = np.add.reduceat((qs >= 0.9).astype(float), starts) if n else np.zeros(0)
    best_other = np.where(rank_sorted == 0, second[group], first[group])
    mean_other = np.where(sizes[group] > 1, (total[group] - qs) / np.maximum(sizes[group] - 1, 1), np.nan)
    X_sorted = np.column_stack([qs, rank_sorted, first[group] - qs, best_other, total[group], n05[group], n09[group], mean_other])
    X = np.empty_like(X_sorted)
    X[order] = X_sorted
    return X.astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    from sklearn.model_selection import GroupKFold

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-test", action="store_true")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    with S.machine_lock():
        val = S.load(args.pairs_root, args.scores_dir, args.context_dir, "validation", extra_dir=args.extra_dir)
        hold = S.load(args.pairs_root, args.scores_dir, args.context_dir, "holdout", extra_dir=args.extra_dir)
        cal = P.fit_calibrator(np.r_[val["logit"], hold["logit"]], np.r_[val["label"], hold["label"]])
        X, y = np.vstack([S.matrix(val, cal), S.matrix(hold, cal)]), np.r_[val["label"], hold["label"]]
        g = np.r_[val["g"], hold["g"] + val["n"]]
        folds = list(GroupKFold(S.POOLED_FOLDS).split(X, y, g))
        test = None if args.no_test else S.load(args.pairs_root, args.scores_dir, args.context_dir, "test",
                                                keep_targets=True, extra_dir=args.extra_dir)
        Xt = None if test is None else S.matrix(test, cal)
        q1, q1_test = np.zeros(len(y)), (None if Xt is None else np.zeros(len(Xt)))
        for tr, te in folds:
            model = S.fit(X[tr], y[tr], args.threads)
            q1[te] = model.predict(X[te])
            if Xt is not None:
                q1_test += model.predict(Xt) / len(folds)
        X2 = np.column_stack([X, aggregates(g, q1)])
        q2 = np.zeros(len(y))
        for tr, te in folds:
            q2[te] = S.fit(X2[tr], y[tr], args.threads).predict(X2[te])

        nv = len(val["label"])

        def best(q):
            qv, qh = q[:nv], q[nv:]
            tv, _ = P.best_threshold(P.prepare_sweep(val["g"], qv, val["label"], val["truth_len"]))
            th, _ = P.best_threshold(P.prepare_sweep(hold["g"], qh, hold["label"], hold["truth_len"]))

            def f(t, owner):
                return (P.evaluate(val, S.decide(val, qv, t, owner))[0]["macro_f05"],
                        P.evaluate(hold, S.decide(hold, qh, t, owner))[0]["macro_f05"])

            grid = np.unique(np.r_[np.linspace(min(tv, th), max(tv, th), 11), tv, th])
            t = float(max(grid, key=lambda x: sum(f(x, True))))
            owner = sum(f(t, True)) >= sum(f(t, False))
            return t, owner, f(t, owner)

        t1, o1, (f1v, f1h) = best(q1)
        t2, o2, (f2v, f2h) = best(q2)
        log(f"stage2: stage 1 validation {f1v:.5f} holdout {f1h:.5f}; stage 2 validation {f2v:.5f} holdout {f2h:.5f} "
            f"(threshold {t2:.4f}, one-owner {o2})")
        report = {"stage1": {"validation": f1v, "holdout": f1h, "threshold": t1}, "validation_oof_macro_f05": f2v,
                  "holdout_oof_macro_f05": f2h, "threshold": t2, "one_owner": bool(o2),
                  "holdout": P.evaluate(hold, S.decide(hold, q2[nv:], t2, o2))[0], "aggregates": list(AGG_NAMES)}
        if test is not None:
            final = S.fit(X2, y, args.threads)
            q2_test = final.predict(np.column_stack([Xt, aggregates(test["g"], q1_test)]))
            pred = S.decide(test, q2_test, t2, o2)
            out_dir = args.out / "output"
            report["test"] = S.write_test(test, pred, args.test_dir, out_dir)
            report["test"]["validator"] = P.run_validator(P.VALIDATOR, out_dir, args.test_dir)
            log(f"stage2: test {report['test']['matched_pairs']:,} matches; validator {report['test']['validator']['passed']}")
        report["seconds"] = time.perf_counter() - started
        atomic_write_json(args.out / "stack_report.json", report)


if __name__ == "__main__":
    main()
