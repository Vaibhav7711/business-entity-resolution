"""Final decision stage: a gradient-boosted stacker over cross-encoder scores and corpus-context features.

Per pairs row the stacker sees the ``ce_policy`` stack features (logit, calibrated p, rank and gap within the S1's
list, number of likely matches in the list, filter score/rank) plus the ``ce_context`` columns (how many S1/targets
share the name/address, empty address, exact equality, list duplicates). It is trained on fold-0 validation only:

1. isotonic calibration of the logit on validation;
2. 3-fold out-of-fold stacker predictions (folds by S1) -> the macro-F0.5-optimal threshold, and whether the
   one-owner rule helps (kept unless it lowers validation);
3. one stacker refit on all of validation -> holdout scored once (paired bootstrap CI vs the plain threshold);
4. test: stacker -> threshold -> one owner across all test S1 -> ``matching_results.tsv`` + ``candidate_pairs.tsv``
   (exactly the scored candidates) -> official validator.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import contextlib
import fcntl
import json
import os
import time
from pathlib import Path

import numpy as np

from . import ce_policy as P
from .evaluate_phase1c import atomic_write_json
from .phase2a_env import log

PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
          "seed": 20260927, "deterministic": True, "force_row_wise": True}
ROUNDS = 600
FOLDS = 3
POOLED_FOLDS = 4
LOCK_PATH = Path(os.environ.get("CE_STACK_LOCK", "/tmp/ce_stack.lock"))   # one stacker at a time per machine


@contextlib.contextmanager
def machine_lock(path: Path = LOCK_PATH):
    """Serialise stackers: LightGBM runs all cores, and concurrent runs spin-wait against each other (10x slower)."""
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"stack: waiting for {path} (another stacker is running)")
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def scored_mask(split_dir: Path, k: int | None) -> np.ndarray:
    import pyarrow.parquet as pq

    ranks = np.concatenate([pq.read_table(p, columns=["filter_rank"]).column("filter_rank").to_numpy()
                            for p in sorted(split_dir.glob("part-*.parquet"))])
    return np.ones(len(ranks), bool) if k is None else ranks < k


def load(pairs_root: Path, scores_dir: Path, context_dir: Path, split: str, keep_targets: bool = False,
         extra_dir: Path | list[Path] | None = None) -> dict:
    k = P.scored_k(scores_dir)
    d = P.load_split(pairs_root, scores_dir, split, keep_targets=keep_targets, max_rank=k)
    with np.load(context_dir / f"{split}.npz", allow_pickle=False) as z:
        ctx, names = z["X"], [str(x) for x in z["names"]]
    extras = [] if extra_dir is None else [extra_dir] if isinstance(extra_dir, Path) else list(extra_dir)
    for directory in extras:
        with np.load(directory / f"{split}.npz", allow_pickle=False) as z:
            if len(z["X"]) != len(ctx):
                raise ValueError(f"{split}: extra features in {directory} have {len(z['X']):,} rows, context {len(ctx):,}")
            ctx, names = np.column_stack([ctx, z["X"]]), names + [str(x) for x in z["names"]]
    keep = scored_mask(pairs_root / split, k)
    if len(ctx) != len(keep):
        raise ValueError(f"{split}: {len(ctx):,} context rows for {len(keep):,} pairs rows")
    d["context"], d["context_names"] = ctx[keep], names
    if len(d["context"]) != len(d["g"]):
        raise ValueError(f"{split}: context/pairs misaligned after the scored-row restriction")
    return d


def matrix(d: dict, cal: dict) -> np.ndarray:
    p = P.calibrate(cal, d["logit"])
    return np.column_stack(list(P.feature_columns(d, p)) + [d["context"]]).astype(np.float32)


def fit(X: np.ndarray, y: np.ndarray, threads: int):
    import lightgbm as lgb

    return lgb.train(PARAMS | {"num_threads": threads}, lgb.Dataset(X, y.astype(np.float32)), ROUNDS)


class Bag:
    """``n`` LightGBM stackers with different seeds and row/column subsampling; predictions are averaged."""

    def __init__(self, models: list):
        self.models = models

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def feature_importance(self, kind: str) -> np.ndarray:
        return np.mean([m.feature_importance(kind) for m in self.models], axis=0)

    def save_model(self, path: str) -> None:
        for i, m in enumerate(self.models):
            m.save_model(path if i == 0 else f"{path}.bag{i}")


def fit_bag(X: np.ndarray, y: np.ndarray, threads: int, n: int):
    import lightgbm as lgb

    if n <= 1:
        return fit(X, y, threads)
    data = lgb.Dataset(X, y.astype(np.float32), free_raw_data=False)
    return Bag([lgb.train(PARAMS | {"num_threads": threads, "seed": PARAMS["seed"] + i, "feature_fraction": 0.8,
                                    "bagging_fraction": 0.8}, data, ROUNDS) for i in range(n)])


def decide(d: dict, q: np.ndarray, t: float, owner: bool) -> np.ndarray:
    pred = q >= t
    return P.one_owner(pred, d, q)[0] if owner else pred


def write_test(test: dict, pred: np.ndarray, test_dir: Path, out_dir: Path) -> dict:
    import pyarrow as pa

    file_ids, countries = P.read_source1(test_dir / "test_source1.tsv")
    position = {s1: i for i, s1 in enumerate(test["s1_ids"])}
    line_s1 = np.fromiter((position.get(x, -1) for x in file_ids), np.int64, len(file_ids))
    run_of_s1 = np.full(test["n"], -1, np.int64)
    run_of_s1[test["run_s1"]] = np.arange(len(test["run_s1"]))
    line_run = np.where(line_s1 >= 0, run_of_s1[np.maximum(line_s1, 0)], -1)
    targets = test["targets"].cast(pa.large_string())
    P.write_tsv(out_dir / "candidate_pairs.tsv", P.CANDIDATE_HEADER, file_ids, line_run,
                P.joined_lists(test["starts"], targets.take(pa.array(test["t_code"]))))
    counts = np.add.reduceat(pred.astype(np.int64), test["starts"][:-1]) if len(pred) else np.zeros(0, np.int64)
    P.write_tsv(out_dir / "matching_results.tsv", P.MATCHING_HEADER, file_ids, line_run,
                P.joined_lists(np.r_[0, np.cumsum(counts)], targets.take(pa.array(test["t_code"][pred]))))
    k = np.bincount(test["g"][pred], minlength=test["n"])
    line_k = np.where(line_s1 >= 0, k[np.maximum(line_s1, 0)], 0)
    labels = np.asarray(countries)
    return {"s1_lines": len(file_ids), "candidate_pairs": int(len(pred)), "matched_pairs": int(pred.sum()),
            "predicted_nonempty_rate": float((line_k > 0).mean()),
            "by_country": {c: {"s1": int((labels == c).sum()), "nonempty": float((line_k[labels == c] > 0).mean()),
                               "mean_set": float(line_k[labels == c].mean())} for c in sorted(set(countries))}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validator", type=Path, default=P.VALIDATOR)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-test", action="store_true", help="tune and report validation/holdout only")
    parser.add_argument("--extra-dir", type=Path, nargs="+",
                        help="extra per-row feature directories (<split>.npz with X, names), e.g. dense_merge, ce_join")
    parser.add_argument("--pooled", action="store_true",
                        help="train on validation and holdout together (out-of-fold threshold), see run_pooled")
    parser.add_argument("--bag", type=int, default=1, help="pooled: average this many seed/subsample stackers")
    args = parser.parse_args(argv)
    with machine_lock():
        (run_pooled if args.pooled else run)(args)


def run(args) -> None:
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    val = load(args.pairs_root, args.scores_dir, args.context_dir, "validation", extra_dir=args.extra_dir)
    cal = P.fit_calibrator(val["logit"], val["label"])
    Xv, yv = matrix(val, cal), val["label"]
    from sklearn.model_selection import GroupKFold

    oof = np.zeros(len(yv))
    for tr, te in GroupKFold(FOLDS).split(Xv, yv, val["g"]):
        oof[te] = fit(Xv[tr], yv[tr], args.threads).predict(Xv[te])
    t, _ = P.best_threshold(P.prepare_sweep(val["g"], oof, yv, val["truth_len"]))
    f_plain = P.evaluate(val, decide(val, oof, t, False))[0]["macro_f05"]
    f_owner = P.evaluate(val, decide(val, oof, t, True))[0]["macro_f05"]
    owner = f_owner >= f_plain
    log(f"stack: threshold {t:.4f}; validation OOF macro F0.5 {max(f_plain, f_owner):.5f} (one-owner {owner})")
    model = fit(Xv, yv, args.threads)
    model.save_model(str(args.out / "stacker.txt"))
    feature_names = ["logit", "p", "logit_rank", "logit_gap", "n_p05", "filter_score", "filter_rank"] + val["context_names"]
    hold = load(args.pairs_root, args.scores_dir, args.context_dir, "holdout", extra_dir=args.extra_dir)
    hq = model.predict(matrix(hold, cal))
    h_metrics, h_f = P.evaluate(hold, decide(hold, hq, t, owner))
    hp = P.calibrate(cal, hold["logit"])
    t_p1, _ = P.best_threshold(P.prepare_sweep(val["g"], P.calibrate(cal, val["logit"]), yv, val["truth_len"]))
    _, base_f = P.evaluate(hold, hp >= t_p1)
    report = {"threshold": float(t), "one_owner": bool(owner), "features": feature_names, "params": PARAMS | {"rounds": ROUNDS},
              "validation_oof_macro_f05": float(max(f_plain, f_owner)), "holdout": h_metrics,
              "holdout_vs_plain_threshold": P.paired_bootstrap(h_f - base_f, 1000, 20260927),
              "gain_share": dict(zip(feature_names, (model.feature_importance("gain") /
                                                     model.feature_importance("gain").sum()).round(4).tolist()))}
    atomic_write_json(args.out / "calibrator.json", cal)
    log(f"stack: holdout macro F0.5 {h_metrics['macro_f05']:.5f} "
        f"(vs plain threshold {report['holdout_vs_plain_threshold']['mean_diff']:+.5f}, "
        f"CI {report['holdout_vs_plain_threshold']['ci95']})")
    del val, hold, Xv
    if not args.no_test:
        test = load(args.pairs_root, args.scores_dir, args.context_dir, "test", keep_targets=True, extra_dir=args.extra_dir)
        tq = model.predict(matrix(test, cal))
        pred = decide(test, tq, t, owner)
        out_dir = args.out / "output"
        report["test"] = write_test(test, pred, args.test_dir, out_dir)
        report["test"]["validator"] = P.run_validator(args.validator, out_dir, args.test_dir)
        log(f"stack: test {report['test']['matched_pairs']:,} matches, non-empty {report['test']['predicted_nonempty_rate']:.3f}; "
            f"validator passed {report['test']['validator']['passed']}")
    report["seconds"] = time.perf_counter() - started
    atomic_write_json(args.out / "stack_report.json", report)
    if not args.no_test and not report["test"]["validator"]["passed"]:
        raise SystemExit("stack: the official validator rejected the test output")


def run_pooled(args) -> None:
    """Validation and holdout pooled (twice the S1 for the stacker): ``POOLED_FOLDS``-fold out-of-fold predictions,
    folds by S1, choose the threshold and the one-owner rule for the sum of both splits' macro F0.5; one stacker
    refit on both scores test. Every reported number is out-of-fold (no S1 is scored by a model that saw it); the
    holdout is no longer untouched, so compare its out-of-fold score with a validation-only stack's holdout score."""
    from sklearn.model_selection import GroupKFold

    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    val = load(args.pairs_root, args.scores_dir, args.context_dir, "validation", extra_dir=args.extra_dir)
    hold = load(args.pairs_root, args.scores_dir, args.context_dir, "holdout", extra_dir=args.extra_dir)
    cal = P.fit_calibrator(np.r_[val["logit"], hold["logit"]], np.r_[val["label"], hold["label"]])
    Xv, Xh = matrix(val, cal), matrix(hold, cal)
    X, y = np.vstack([Xv, Xh]), np.r_[val["label"], hold["label"]]
    groups = np.r_[val["g"], hold["g"] + val["n"]]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(POOLED_FOLDS).split(X, y, groups):
        oof[te] = fit_bag(X[tr], y[tr], args.threads, getattr(args, "bag", 1)).predict(X[te])
    qv, qh = oof[:len(Xv)], oof[len(Xv):]
    tv, _ = P.best_threshold(P.prepare_sweep(val["g"], qv, val["label"], val["truth_len"]))
    th, _ = P.best_threshold(P.prepare_sweep(hold["g"], qh, hold["label"], hold["truth_len"]))

    def scores(t: float, owner: bool) -> tuple[float, float]:
        return (P.evaluate(val, decide(val, qv, t, owner))[0]["macro_f05"],
                P.evaluate(hold, decide(hold, qh, t, owner))[0]["macro_f05"])

    grid = np.unique(np.r_[np.linspace(min(tv, th), max(tv, th), 11), tv, th])
    t = float(max(grid, key=lambda x: sum(scores(x, True))))
    owner = sum(scores(t, True)) >= sum(scores(t, False))
    fv, fh = scores(t, owner)
    h_metrics, _ = P.evaluate(hold, decide(hold, qh, t, owner))
    log(f"stack (pooled): threshold {t:.4f}; out-of-fold macro F0.5 validation {fv:.5f}, holdout {fh:.5f} "
        f"(one-owner {owner})")
    model = fit_bag(X, y, args.threads, getattr(args, "bag", 1))
    model.save_model(str(args.out / "stacker.txt"))
    feature_names = ["logit", "p", "logit_rank", "logit_gap", "n_p05", "filter_score", "filter_rank"] + val["context_names"]
    report = {"mode": "pooled validation+holdout", "bag": getattr(args, "bag", 1), "threshold": t, "one_owner": bool(owner), "features": feature_names,
              "params": PARAMS | {"rounds": ROUNDS, "folds": POOLED_FOLDS},
              "validation_oof_macro_f05": float(fv), "holdout_oof_macro_f05": float(fh), "holdout": h_metrics,
              "gain_share": dict(zip(feature_names, (model.feature_importance("gain") /
                                                     model.feature_importance("gain").sum()).round(4).tolist()))}
    atomic_write_json(args.out / "calibrator.json", cal)
    del val, hold, X, Xv, Xh
    if not args.no_test:
        test = load(args.pairs_root, args.scores_dir, args.context_dir, "test", keep_targets=True, extra_dir=args.extra_dir)
        pred = decide(test, model.predict(matrix(test, cal)), t, owner)
        out_dir = args.out / "output"
        report["test"] = write_test(test, pred, args.test_dir, out_dir)
        report["test"]["validator"] = P.run_validator(args.validator, out_dir, args.test_dir)
        log(f"stack (pooled): test {report['test']['matched_pairs']:,} matches; validator passed "
            f"{report['test']['validator']['passed']}")
    report["seconds"] = time.perf_counter() - started
    atomic_write_json(args.out / "stack_report.json", report)
    if not args.no_test and not report["test"]["validator"]["passed"]:
        raise SystemExit("stack: the official validator rejected the test output")


if __name__ == "__main__":
    main()
