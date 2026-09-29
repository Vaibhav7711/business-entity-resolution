"""CE policy: cross-encoder logits -> calibrated P(match) -> per-S1 match sets -> submission files.

Run from code/business_entity_resolution:

    python3 -m src.ce_policy --pairs-root <pairs root> --scores-dir <ce_out>/scores \
        --test-dir <test TSV dir> --out <dir> --stage all

Stages (``--stage all`` runs both):

* ``tune``: fold-0 ``validation`` decides everything. An isotonic calibrator maps sigmoid(logit) to
  P(match). Rule families are tuned on the official macro F0.5 (per S1 with ``truth_len`` from
  ``s1.parquet``, so gold that blocking missed, singletons, and S1 without candidates all count):

  - P1 ``threshold``: rows with p >= t. Every distinct p is tried in one sorted sweep: rows of an S1
    enter in descending p, so running sums of per-row changes in the S1's F0.5 are exact at each
    block of equal p;
  - P2 ``empty_guard``: P1, but nothing for an S1 whose best p is below t_empty (t_empty >= t);
  - P3 ``relative``: rows with p >= t and p >= r * best p of the S1;
  - P4 one-owner: after P1/P2/P3, a target claimed by several S1 stays only with the highest p (then
    higher logit, then lower filter rank); t is re-tuned on a window around the base optimum.

  A stacked variant (logistic regression on per-pair features, 2-fold group CV by S1 on validation,
  calibrator refit inside each fold) is tuned the same way on its out-of-fold scores and competes only
  if it beats the best plain policy. The final policy is the best validation macro F0.5; within
  ``--tie-tolerance`` the simpler one wins. ``holdout`` is scored after the choice is fixed and never
  influences it; a paired S1 bootstrap gives the CI of chosen minus P1.
* ``write``: loads the chosen policy and calibrator from ``--out``, applies them to every test pair
  (one-owner then resolves conflicts across all test S1), writes ``matching_results.tsv`` and
  ``candidate_pairs.tsv`` in ``test_source1.tsv`` order, and runs the official validator.

Row identity is positional: ``<scores-dir>/<split>.npy`` holds one logit per pairs row in concatenated
part order. Rows of an S1 are contiguous (checked: every run is a distinct S1) and (S1, target) pairs
are unique (checked). Target IDs are only dictionary-encoded, never merged by text.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json
from .phase2a_env import log

FEATURES = ("logit", "p", "logit_rank", "logit_gap", "n_p05", "filter_score", "filter_rank")
RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
GUARD_QUANTILES = 401
OWNER_WINDOW = np.linspace(-0.2, 0.2, 41)
NOTHING = 2.0                       # threshold above every probability: predict nothing
COMPLEXITY = {"threshold": 0, "empty_guard": 1, "relative": 1}
SHORT = {"threshold": "P1", "empty_guard": "P2", "relative": "P3"}
CHUNK = 5_000_000
WRITE_CHUNK = 100_000
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids"
VALIDATOR = ROOT / "student_resource" / "utils" / "validate_submission.py"
TRAIN_DIR = ROOT / "student_resource" / "dataset" / "train"


# ---------------------------------------------------------------------------
# Inputs


def scored_k(scores_dir: Path) -> int | None:
    """K the cross-encoder run scored on test (``<ce_out>/test_k.json``); None when every row was scored."""
    path = Path(scores_dir).parent / "test_k.json"
    return int(json.loads(path.read_text())["k"]) if path.exists() else None


def restrict(d: dict, keep: np.ndarray) -> dict:
    """Only rows with ``keep`` (the candidates the cross-encoder scored); S1 runs recomputed, truth_len unchanged."""
    out = dict(d)
    for name in ("g", "label", "logit", "t_code", "filter_score", "filter_rank"):
        if out.get(name) is not None:
            out[name] = out[name][keep]
    rows = len(out["g"])
    first = np.ones(rows, bool)
    if rows:
        first[1:] = out["g"][1:] != out["g"][:-1]
    out["starts"] = np.r_[np.flatnonzero(first), rows].astype(np.int64)
    out["run_s1"] = out["g"][out["starts"][:-1]].astype(np.int64)
    out["checks"] = dict(d["checks"], rows_scored=int(rows), rows_dropped_unscored=int((~keep).sum()))
    return out


def load_split(pairs_root: Path, scores_dir: Path, split: str, keep_targets: bool = False,
               max_rank: int | None = None) -> dict:
    """One split as flat row arrays in pairs order: S1 runs ``starts`` / ``run_s1``, ``g`` (row -> index in
    ``s1.parquet``), ``t_code`` (row -> index in the split's unified ``t_id`` dictionary, kept as ``targets``
    when asked), logits, and ``label`` (None on test)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    directory = pairs_root / split
    s1 = pq.read_table(directory / "s1.parquet", columns=["s1_id", "truth_len", "n_cand", "retrieved_truth"])
    s1_ids = s1.column("s1_id").to_pylist()
    position = {s1_id: i for i, s1_id in enumerate(s1_ids)}
    if len(position) != len(s1_ids):
        raise ValueError(f"{split}: duplicate s1_id in s1.parquet")
    run_ids, firsts, targets, previous = [], [], [], None
    columns = {name: [] for name in ("label", "filter_score", "filter_rank")}
    for path in sorted(directory.glob("part-*.parquet")):
        table = pq.read_table(path, columns=["s1_id", "t_id", *columns], read_dictionary=["t_id"])
        if table.num_rows == 0:
            continue
        ids = table.column("s1_id").combine_chunks()
        if ids.null_count or table.column("t_id").null_count:
            raise ValueError(f"{path}: null ids")
        first = np.ones(len(ids), bool)
        first[1:] = pc.not_equal(ids.slice(1), ids.slice(0, len(ids) - 1)).to_numpy(zero_copy_only=False)
        first[0] = ids[0].as_py() != previous
        previous = ids[-1].as_py()
        run_ids += pc.filter(ids, pa.array(first)).to_pylist()
        firsts.append(first)
        for name, values in columns.items():
            values.append(table.column(name).to_numpy())
        targets += table.column("t_id").chunks
        del table, ids
    first = np.concatenate(firsts) if firsts else np.zeros(0, bool)
    rows = len(first)
    starts = np.r_[np.flatnonzero(first), rows].astype(np.int64)
    run_s1 = np.fromiter((position.get(x, -1) for x in run_ids), np.int64, len(run_ids))
    if (run_s1 < 0).any():
        raise ValueError(f"{split}: {int((run_s1 < 0).sum()):,} pairs S1 are missing from s1.parquet")
    if len(np.unique(run_s1)) != len(run_s1):
        raise ValueError(f"{split}: rows of an S1 are not contiguous")
    g = np.repeat(run_s1.astype(np.int32), np.diff(starts))
    dictionary, t_code = pa.array([], pa.string()), np.zeros(0, np.int32)
    if targets:
        unified = pa.Table.from_arrays([pa.chunked_array(targets)], names=["t"]).unify_dictionaries().column("t")
        dictionary = unified.chunk(0).dictionary
        t_code = np.concatenate([chunk.indices.to_numpy() for chunk in unified.chunks]).astype(np.int32, copy=False)
        del unified
    del targets
    key = g.astype(np.int64)
    key *= max(len(dictionary), 1)
    key += t_code
    key.sort()
    if rows and (key[1:] == key[:-1]).any():
        raise ValueError(f"{split}: duplicate (s1_id, t_id) pairs")
    del key
    raw = np.load(scores_dir / f"{split}.npy", mmap_mode="r")
    if raw.shape != (rows,):
        raise ValueError(f"{split}: scores shape {raw.shape} for {rows:,} pairs rows")
    logit = np.array(raw, np.float32)
    frank = np.concatenate(columns["filter_rank"]) if rows else np.zeros(0, np.int16)
    keep = frank < max_rank if max_rank is not None else np.ones(rows, bool)
    if not np.isfinite(logit[keep]).all():
        raise ValueError(f"{split}: non-finite logits among the scored rows")
    truth = s1.column("truth_len").to_numpy().astype(np.int64)
    label = np.concatenate(columns["label"]) if rows else np.zeros(0, np.int8)
    labelled = bool(len(truth) == 0 or truth.min() >= 0)
    n = len(s1_ids)
    n_rows_s1 = np.bincount(g, minlength=n)
    checks = {"rows": rows, "s1": n, "s1_without_rows": int((n_rows_s1 == 0).sum()),
              "n_cand_mismatch": int((s1.column("n_cand").to_numpy() != n_rows_s1).sum()),
              "distinct_targets": int(len(np.unique(t_code)))}
    if labelled:
        if rows and not np.isin(label, (0, 1)).all():
            raise ValueError(f"{split}: labels outside {{0, 1}}")
        label = label.astype(bool)
        in_list = np.bincount(g[label], minlength=n)
        if (in_list > truth).any():
            raise ValueError(f"{split}: more labelled matches in a list than truth_len")
        checks |= {"singletons": int((truth == 0).sum()), "truth_total": int(truth.sum()),
                   "truth_in_lists": int(in_list.sum()),
                   "retrieved_truth_mismatch": int((s1.column("retrieved_truth").to_numpy() != in_list).sum())}
    else:
        label = None
    out = {"split": split, "n": n, "s1_ids": s1_ids, "truth_len": truth, "starts": starts, "run_s1": run_s1, "g": g,
           "label": label, "logit": logit, "t_code": t_code, "checks": checks,
           "filter_score": np.concatenate(columns["filter_score"]) if rows else np.zeros(0, np.float32),
           "filter_rank": np.concatenate(columns["filter_rank"]) if rows else np.zeros(0, np.int16)}
    if keep_targets:
        out["targets"] = dictionary
    if not keep.all():
        out = restrict(out, keep)
        log(f"{split}: kept the {int(keep.sum()):,} rows with filter_rank < {max_rank} (the scored candidates)")
    log(f"{split}: {rows:,} rows, {n:,} S1 ({checks['s1_without_rows']:,} without candidates)")
    return out


def read_source1(path: Path) -> tuple[list[str], list[str]]:
    """entity_id and country of every row in file order, parsed like the pipeline (csv.DictReader)."""
    ids, countries = [], []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            ids.append(row["entity_id"])
            countries.append(row["country"])
    return ids, countries


def train_countries(train_dir: Path | None, wanted: set[str]) -> dict[str, str] | None:
    path = train_dir / "train_source1.tsv" if train_dir else None
    if path is None or not path.exists():
        return None
    ids, countries = read_source1(path)
    return {i: c for i, c in zip(ids, countries) if i in wanted}


# ---------------------------------------------------------------------------
# Metric


def f05(tp: np.ndarray, k: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-S1 F0.5 from true positives, predicted count, and truth_len (= 1.25 tp / (k + 0.25 truth))."""
    tp, k, truth = (np.asarray(x, np.float64) for x in (tp, k, truth))
    value = np.where(tp > 0, 1.25 * tp / np.maximum(k + 0.25 * truth, 1e-12), 0.0)
    return np.where(truth == 0, (k == 0).astype(np.float64), value)


def evaluate(d: dict, pred: np.ndarray) -> tuple[dict, np.ndarray]:
    """Macro F0.5 over every S1 of the split, with per-S1 values (for the bootstrap and slices)."""
    n, truth = d["n"], d["truth_len"]
    k = np.bincount(d["g"][pred], minlength=n)
    tp = np.bincount(d["g"][pred & d["label"]], minlength=n)
    f = f05(tp, k, truth)
    has_pred, has_truth, single = k > 0, truth > 0, truth == 0
    metrics = {
        "macro_f05": float(f.mean()) if n else 0.0,
        "mean_precision": float((tp[has_pred] / k[has_pred]).mean()) if has_pred.any() else None,
        "mean_recall": float((tp[has_truth] / truth[has_truth]).mean()) if has_truth.any() else None,
        "singleton_accuracy": float((k[single] == 0).mean()) if single.any() else None,
        "predicted_nonempty_rate": float(has_pred.mean()) if n else 0.0,
        "predicted_pairs": int(k.sum()), "true_positive_pairs": int(tp.sum()), "s1": int(n)}
    return metrics, f


def prepare_sweep(g: np.ndarray, score: np.ndarray, label: np.ndarray, truth_len: np.ndarray) -> dict:
    """Per-row change of its S1's F0.5 when the row joins the predicted set, rows by descending score.
    Rows of one S1 join in descending score, so the running sum at the end of each block of equal scores
    is exactly the numerator of 'predict score >= that value' (the per-S1 sums telescope)."""
    order = np.lexsort((-score, g))
    gs = g[order]
    first = np.ones(len(gs), bool)
    first[1:] = gs[1:] != gs[:-1]
    start = np.flatnonzero(first)
    run = np.cumsum(first) - 1
    k = np.arange(1, len(gs) + 1) - start[run]
    hits = label[order].astype(np.int64)
    cum = np.cumsum(hits)
    tp = cum - (cum - hits)[start][run]
    truth = truth_len[gs]
    f = f05(tp, k, truth)
    before = np.empty_like(f)
    before[1:] = f[:-1]
    before[first] = truth[first] == 0
    values = score[order]
    down = np.argsort(-values, kind="stable")
    values = values[down]
    end = np.ones(len(values), bool)
    end[:-1] = values[1:] != values[:-1]
    return {"delta": (f - before)[down], "g": gs[down], "ends": np.flatnonzero(end),
            "thresholds": np.r_[NOTHING, values[end]], "base": int((truth_len == 0).sum()), "n": len(truth_len)}


def curve(prep: dict, mask: np.ndarray | None = None) -> np.ndarray:
    """Macro F0.5 at every ``prep['thresholds']`` (descending; index 0 predicts nothing). ``mask`` (per
    sorted row) removes rows, leaving their S1 at the empty-prediction score."""
    delta = prep["delta"] if mask is None else np.where(mask, prep["delta"], 0.0)
    total = np.cumsum(delta)[prep["ends"]]
    return (prep["base"] + np.r_[0.0, total]) / max(prep["n"], 1)


def best_threshold(prep: dict, mask: np.ndarray | None = None, cap: float | None = None) -> tuple[float, float]:
    """Best (threshold, macro F0.5); ties go to the higher threshold. ``cap`` limits thresholds to <= cap."""
    macro = curve(prep, mask)
    if cap is not None:
        allowed = prep["thresholds"] <= cap
        allowed[0] = True
        macro = np.where(allowed, macro, -1.0)
    i = int(np.argmax(macro))
    return float(prep["thresholds"][i]), float(macro[i])


# ---------------------------------------------------------------------------
# Scores


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -40.0, 40.0)))


def fit_calibrator(logit: np.ndarray, label: np.ndarray) -> dict:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(sigmoid(logit), label.astype(np.float64))
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def calibrate(cal: dict, logit: np.ndarray) -> np.ndarray:
    """Isotonic prediction (linear between knots, clipped outside), streamed in chunks."""
    x, y = np.asarray(cal["x"], np.float64), np.asarray(cal["y"], np.float64)
    out = np.empty(len(logit), np.float64)
    for s in range(0, len(logit), CHUNK):
        out[s:s + CHUNK] = np.interp(sigmoid(logit[s:s + CHUNK]), x, y)
    return out


def run_max(d: dict, x: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(x, d["starts"][:-1]) if len(x) else np.zeros(0, x.dtype)


def per_row(d: dict, per_run: np.ndarray) -> np.ndarray:
    return np.repeat(per_run, np.diff(d["starts"]))


def row_rank(d: dict, x: np.ndarray) -> np.ndarray:
    """0 = highest ``x`` within the S1 (ties: lower filter rank first)."""
    lengths = np.diff(d["starts"])
    order = np.lexsort((d["filter_rank"], -x, np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)))
    position = np.arange(len(x), dtype=np.int64)
    position -= np.repeat(d["starts"][:-1], lengths)
    rank = np.empty(len(x), np.float64)
    rank[order] = position
    return rank


def feature_columns(d: dict, p: np.ndarray):
    """The stacked model's per-pair features in ``FEATURES`` order, one column at a time (bounded memory)."""
    logit = d["logit"].astype(np.float64)
    yield logit
    yield p
    yield row_rank(d, d["logit"])
    yield logit - per_row(d, run_max(d, logit))
    count = np.add.reduceat((p >= 0.5).astype(np.int64), d["starts"][:-1]) if len(p) else np.zeros(0, np.int64)
    yield per_row(d, count).astype(np.float64)
    yield d["filter_score"].astype(np.float64)
    yield d["filter_rank"].astype(np.float64)


def fit_stack(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression

    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    model = LogisticRegression(max_iter=1000).fit((X - mean) / scale, y)
    return {"features": list(FEATURES), "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}


def stack_probability(model: dict, columns, n: int) -> np.ndarray:
    weights = np.asarray(model["coef"]) / np.asarray(model["scale"])
    z = np.full(n, model["intercept"] - float(weights @ np.asarray(model["mean"])))
    for column, weight in zip(columns, weights):
        z += weight * column
    return sigmoid(z)


def stacked_oof(d: dict) -> np.ndarray:
    """Out-of-fold stacked probabilities on validation: 2 folds by S1, calibrator and model fit per fold."""
    from sklearn.model_selection import GroupKFold

    q = np.zeros(len(d["logit"]))
    for fit_rows, score_rows in GroupKFold(n_splits=2).split(d["logit"], groups=d["g"]):
        cal = fit_calibrator(d["logit"][fit_rows], d["label"][fit_rows])
        X = np.column_stack(list(feature_columns(d, calibrate(cal, d["logit"]))))
        model = fit_stack(X[fit_rows], d["label"][fit_rows])
        q[score_rows] = stack_probability(model, X[score_rows].T, len(score_rows))
    return q


# ---------------------------------------------------------------------------
# Policies


def make_policy(source: str, family: str, params: dict, one_owner: bool = False) -> dict:
    name = f"P4({SHORT[family]})" if one_owner else SHORT[family]
    return {"name": f"stacked-{name}" if source == "stacked" else name, "source": source, "family": family,
            "params": params, "one_owner": one_owner,
            "complexity": COMPLEXITY[family] + int(one_owner) + 2 * (source == "stacked")}


def one_owner(pred: np.ndarray, d: dict, score: np.ndarray) -> tuple[np.ndarray, dict]:
    """Each target keeps only its best claim: highest score, then higher logit, then lower filter rank."""
    rows = np.flatnonzero(pred)
    t = d["t_code"][rows]
    order = np.lexsort((d["g"][rows], d["filter_rank"][rows], -d["logit"][rows], -score[rows], t))
    t_sorted = t[order]
    lose = np.zeros(len(order), bool)
    lose[1:] = t_sorted[1:] == t_sorted[:-1]
    losers = rows[order[lose]]
    out = pred.copy()
    out[losers] = False
    stats = {"claims": int(len(rows)), "targets_in_conflict": int(len(np.unique(d["t_code"][losers]))),
             "claims_dropped": int(len(losers))}
    if d["label"] is not None:
        stats["dropped_true"] = int(d["label"][losers].sum())
        stats["dropped_false"] = stats["claims_dropped"] - stats["dropped_true"]
    return out, stats


def apply_policy(spec: dict, d: dict, score: np.ndarray) -> tuple[np.ndarray, dict | None]:
    params = spec["params"]
    pred = score >= params["t"]
    if spec["family"] == "empty_guard":
        pred &= per_row(d, run_max(d, score)) >= params["t_empty"]
    elif spec["family"] == "relative":
        pred &= score >= params["r"] * per_row(d, run_max(d, score))
    if spec["one_owner"]:
        return one_owner(pred, d, score)
    return pred, None


def tune_one_owner(d: dict, score: np.ndarray, base: dict) -> dict:
    params = base["params"]
    grid = np.unique(np.r_[params["t"], np.clip(params["t"] + OWNER_WINDOW, 0.0, 1.0)])[::-1]
    if base["family"] == "empty_guard":
        grid = grid[grid <= params["t_empty"]] if (grid <= params["t_empty"]).any() else grid[-1:]
    best = None
    for t in grid:
        spec = make_policy(base["source"], base["family"], params | {"t": float(t)}, one_owner=True)
        pred, stats = apply_policy(spec, d, score)
        metrics, _ = evaluate(d, pred)
        if best is None or metrics["macro_f05"] > best["validation"]["macro_f05"]:
            best = spec | {"base": base["name"], "validation": metrics, "validation_one_owner": stats}
    return best


def tune_family(d: dict, score: np.ndarray, source: str) -> list[dict]:
    """P1-P3 tuned by exact sweeps, then P4 on each; every chosen value re-checked by direct evaluation."""
    started = time.perf_counter()
    prep = prepare_sweep(d["g"], score, d["label"], d["truth_len"])
    t, value = best_threshold(prep)
    found = [(make_policy(source, "threshold", {"t": t}), value)]
    maxima = run_max(d, score)
    s1_max = np.full(d["n"], -np.inf)
    s1_max[d["run_s1"]] = maxima
    sorted_max = s1_max[prep["g"]]
    best = (-1.0, None)
    guards = np.unique(np.quantile(maxima, np.linspace(0, 1, GUARD_QUANTILES))) if len(maxima) else []
    for guard in guards:
        t, value = best_threshold(prep, sorted_max >= guard, cap=float(guard))
        if value > best[0]:
            best = (value, {"t": t, "t_empty": float(guard)})
    found.append((make_policy(source, "empty_guard", best[1] or {"t": NOTHING, "t_empty": NOTHING}), best[0]))
    rowmax = per_row(d, maxima)
    best = (-1.0, None)
    for r in RATIOS:
        keep = score >= r * rowmax
        t, value = best_threshold(prepare_sweep(d["g"][keep], score[keep], d["label"][keep], d["truth_len"]))
        if value > best[0]:
            best = (value, {"t": t, "r": r})
    found.append((make_policy(source, "relative", best[1]), best[0]))
    policies = []
    for spec, value in found:
        metrics, _ = evaluate(d, apply_policy(spec, d, score)[0])
        if abs(metrics["macro_f05"] - value) > 1e-9:
            raise RuntimeError(f"{spec['name']}: sweep {value:.10f} != direct {metrics['macro_f05']:.10f}")
        policies.append(spec | {"validation": metrics})
    policies += [tune_one_owner(d, score, base) for base in list(policies)]
    for p in policies:
        log(f"tune {p['name']}: {p['params']} -> validation macro F0.5 {p['validation']['macro_f05']:.5f}")
    log(f"tune {source}: {time.perf_counter() - started:,.1f}s")
    return policies


def choose(policies: list[dict], tie: float) -> dict:
    best = max(p["validation"]["macro_f05"] for p in policies)
    near = [p for p in policies if p["validation"]["macro_f05"] >= best - tie]
    return min(near, key=lambda p: (p["complexity"], -p["validation"]["macro_f05"]))


def paired_bootstrap(diff: np.ndarray, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    means = []
    for done in range(0, reps, 50):
        idx = rng.integers(0, len(diff), size=(min(50, reps - done), len(diff)))
        means.append(diff[idx].mean(axis=1))
    means = np.concatenate(means)
    return {"reps": reps, "seed": seed, "mean_diff": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "share_reps_le_0": float((means <= 0).mean())}


def pair_auc(d: dict) -> float | None:
    from sklearn.metrics import roc_auc_score

    if d["label"].all() or not d["label"].any():
        return None
    return float(roc_auc_score(d["label"], d["logit"]))


def country_slices(d: dict, countries: dict | None, values: dict[str, np.ndarray]) -> dict | None:
    if countries is None:
        return None
    labels = np.asarray([countries.get(s1, "unknown") for s1 in d["s1_ids"]])
    out = {}
    for country in sorted(set(labels.tolist()), key=lambda c: -int((labels == c).sum())):
        mask = labels == country
        out[country] = {"s1": int(mask.sum())} | {name: float(f[mask].mean()) for name, f in values.items()}
    return out


# ---------------------------------------------------------------------------
# Stage: tune


def tune(args) -> dict:
    started = time.perf_counter()
    out = args.out
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    k_scored = scored_k(args.scores_dir)
    val = load_split(args.pairs_root, args.scores_dir, "validation", max_rank=k_scored)
    cal = fit_calibrator(val["logit"], val["label"])
    cal_file = cal | {"fit_split": "validation", "rows": len(val["logit"]), "input": "sigmoid(logit)",
                      "interpolation": "linear, clipped", "run_id": run_id}
    val_p = calibrate(cal, val["logit"])
    log(f"calibrator: {len(cal['x']):,} knots from {len(val['logit']):,} validation pairs")
    plain = tune_family(val, val_p, "ce")
    val_q = stacked_oof(val)
    stacked = tune_family(val, val_q, "stacked")
    model = fit_stack(np.column_stack(list(feature_columns(val, val_p))), val["label"])
    best_plain = max(p["validation"]["macro_f05"] for p in plain)
    best_stacked = max(p["validation"]["macro_f05"] for p in stacked)
    included = best_stacked > best_plain + args.tie_tolerance
    for p in plain + stacked:
        p["eligible"] = p["source"] == "ce" or included
    chosen = choose([p for p in plain + stacked if p["eligible"]], args.tie_tolerance)
    baseline = plain[0]
    log(f"chosen {chosen['name']} {chosen['params']} (validation {chosen['validation']['macro_f05']:.5f}; "
        f"P1 {baseline['validation']['macro_f05']:.5f}; stacked best {best_stacked:.5f}, included {included})")
    val_scores = {"ce": val_p, "stacked": val_q}
    val_f = {name: evaluate(val, apply_policy(spec, val, val_scores[spec["source"]])[0])[1]
             for name, spec in (("chosen", chosen), ("p1", baseline))}
    val_ids, val_auc, val_oracle = val["s1_ids"], pair_auc(val), evaluate(val, val["label"])[0]["macro_f05"]
    val_checks = val["checks"]
    del val, val_p, val_q, val_scores

    hold = load_split(args.pairs_root, args.scores_dir, "holdout", max_rank=k_scored)
    hold_p = calibrate(cal, hold["logit"])
    hold_scores = {"ce": hold_p, "stacked": stack_probability(model, feature_columns(hold, hold_p), len(hold_p))}
    hold_f = {}
    for spec in plain + stacked:
        pred, stats = apply_policy(spec, hold, hold_scores[spec["source"]])
        spec["holdout"], f = evaluate(hold, pred)
        if stats:
            spec["holdout_one_owner"] = stats
        if spec is chosen:
            hold_f["chosen"] = f
        if spec is baseline:
            hold_f["p1"] = f
    boot = paired_bootstrap(hold_f["chosen"] - hold_f["p1"], args.bootstrap_reps, args.seed)
    log(f"holdout: chosen {chosen['holdout']['macro_f05']:.5f}, P1 {baseline['holdout']['macro_f05']:.5f}, "
        f"diff CI95 [{boot['ci95'][0]:+.5f}, {boot['ci95'][1]:+.5f}]")
    countries = train_countries(args.train_dir, set(val_ids) | set(hold["s1_ids"]))
    by_name = {p["name"]: p for p in plain + stacked}
    owner = {p["name"]: {"validation_delta": p["validation"]["macro_f05"] - by_name[p["base"]]["validation"]["macro_f05"],
                         "holdout_delta": p["holdout"]["macro_f05"] - by_name[p["base"]]["holdout"]["macro_f05"],
                         "validation_conflicts": p["validation_one_owner"], "holdout_conflicts": p["holdout_one_owner"]}
             for p in plain + stacked if p["one_owner"]}
    spec_keys = ("name", "source", "family", "params", "one_owner", "complexity")
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {"pairs_root": str(args.pairs_root), "scores_dir": str(args.scores_dir),
                   "validation": val_checks, "holdout": hold["checks"]},
        "calibrator": {"fit_split": "validation", "knots": len(cal["x"]), "run_id": run_id},
        "pair_auc": {"validation": val_auc, "holdout": pair_auc(hold)},
        "oracle_macro_f05": {"validation": val_oracle, "holdout": evaluate(hold, hold["label"])[0]["macro_f05"],
                             "meaning": "predict exactly the true pairs in each candidate list (ceiling of any rule)"},
        "selection": {"rule": "best validation macro F0.5; within tie_tolerance the lower complexity wins; holdout "
                              "never used", "tie_tolerance": args.tie_tolerance, "best_plain_validation": best_plain,
                      "best_stacked_validation_cv": best_stacked, "stacked_included": included},
        "policies": plain + stacked,
        "chosen": {k: chosen[k] for k in spec_keys} | ({"model": model} if chosen["source"] == "stacked" else {})
                  | {"validation": chosen["validation"], "holdout": chosen["holdout"]},
        "baseline": baseline["name"],
        "holdout_bootstrap_chosen_minus_p1": boot,
        "one_owner": owner,
        "stacked_model": model,
        "countries": {"validation": country_slices({"s1_ids": val_ids}, countries, val_f),
                      "holdout": country_slices(hold, countries, hold_f)},
        "seconds": time.perf_counter() - started}
    atomic_write_json(out / "calibrator.json", cal_file)
    atomic_write_json(out / "policy_report.json", report)
    write_markdown(out, report)
    return report


# ---------------------------------------------------------------------------
# Stage: write


def joined_lists(offsets: np.ndarray, values):
    """One comma-joined string per S1 run from flat target IDs and run offsets."""
    import pyarrow as pa
    import pyarrow.compute as pc

    lists = pa.LargeListArray.from_arrays(pa.array(np.asarray(offsets, np.int64)), values)
    return pc.binary_join(lists, pa.scalar(",", pa.large_string()))


def write_tsv(path: Path, header: str, ids: list[str], line_run: np.ndarray, joined) -> None:
    import pyarrow as pa

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        file.write(header + "\n")
        for s in range(0, len(ids), WRITE_CHUNK):
            runs = line_run[s:s + WRITE_CHUNK]
            has = runs >= 0
            texts = joined.take(pa.array(np.where(has, runs, 0))).to_pylist() if len(joined) else [""] * len(runs)
            file.write("".join(f"{i}\t{t if h else ''}\n" for i, t, h in zip(ids[s:s + WRITE_CHUNK], texts, has.tolist())))
    os.replace(temporary, path)


def write_outputs(spec: dict, cal: dict, args) -> dict:
    import pyarrow as pa

    test = load_split(args.pairs_root, args.scores_dir, "test", keep_targets=True, max_rank=scored_k(args.scores_dir))
    file_ids, countries = read_source1(args.test_dir / "test_source1.tsv")
    if len(set(file_ids)) != len(file_ids):
        raise ValueError("duplicate entity_id in test_source1.tsv")
    p = calibrate(cal, test["logit"])
    score = p if spec["source"] == "ce" else stack_probability(spec["model"], feature_columns(test, p), len(p))
    base, _ = apply_policy(spec | {"one_owner": False}, test, score)
    resolved, owner = one_owner(base, test, score)
    pred = resolved if spec["one_owner"] else base
    del p, score, resolved, base
    for name in ("logit", "filter_score", "filter_rank"):
        test.pop(name)
    position = {s1: i for i, s1 in enumerate(test["s1_ids"])}
    line_s1 = np.fromiter((position.get(x, -1) for x in file_ids), np.int64, len(file_ids))
    run_of_s1 = np.full(test["n"], -1, np.int64)
    run_of_s1[test["run_s1"]] = np.arange(len(test["run_s1"]))
    line_run = np.where(line_s1 >= 0, run_of_s1[np.maximum(line_s1, 0)], -1) if test["n"] else np.full(len(file_ids), -1)
    missing_runs = int(len(np.setdiff1d(test["run_s1"], line_s1[line_s1 >= 0])))
    if missing_runs:
        log(f"WARNING: {missing_runs:,} S1 with candidates are not in test_source1.tsv (not written)")
    out_dir = args.out / "output"
    targets = test["targets"].cast(pa.large_string())
    write_tsv(out_dir / "candidate_pairs.tsv", CANDIDATE_HEADER, file_ids, line_run,
              joined_lists(test["starts"], targets.take(pa.array(test["t_code"]))))
    counts = np.add.reduceat(pred.astype(np.int64), test["starts"][:-1]) if len(pred) else np.zeros(0, np.int64)
    write_tsv(out_dir / "matching_results.tsv", MATCHING_HEADER, file_ids, line_run,
              joined_lists(np.r_[0, np.cumsum(counts)], targets.take(pa.array(test["t_code"][pred]))))
    k = np.bincount(test["g"][pred], minlength=test["n"])
    n_cand = np.bincount(test["g"], minlength=test["n"])
    line_k = np.where(line_s1 >= 0, k[np.maximum(line_s1, 0)], 0) if test["n"] else np.zeros(len(file_ids), np.int64)
    line_c = np.where(line_s1 >= 0, n_cand[np.maximum(line_s1, 0)], 0) if test["n"] else np.zeros(len(file_ids), np.int64)
    labels = np.asarray(countries)
    by_country = {}
    for country in sorted(set(countries), key=lambda c: -int((labels == c).sum())):
        mask = labels == country
        by_country[country] = {"s1": int(mask.sum()), "predicted_nonempty_rate": float((line_k[mask] > 0).mean()),
                               "mean_set_size": float(line_k[mask].mean()), "mean_candidates": float(line_c[mask].mean())}
    return {"s1_lines": len(file_ids), "candidate_pairs": int(len(pred)), "matched_pairs": int(pred.sum()),
            "predicted_nonempty_rate": float((line_k > 0).mean()) if len(line_k) else 0.0,
            "mean_set_size": float(line_k.mean()) if len(line_k) else 0.0,
            "one_owner": {"applied": bool(spec["one_owner"])} | owner,
            "file_s1_without_pairs_entry": int((line_s1 < 0).sum()), "pairs_s1_not_in_file": missing_runs,
            "checks": test["checks"], "by_country": by_country,
            "files": {"matching": str(out_dir / "matching_results.tsv"), "candidate": str(out_dir / "candidate_pairs.tsv")}}


def run_validator(validator: Path, out_dir: Path, test_dir: Path) -> dict:
    """Official validator with ``--check-ids`` (when supported) and the candidate file; if that run dies
    without a verdict (memory), once more without the candidate cross-check."""
    extra = ["--check-ids"] if "--check-ids" in validator.read_text() else []
    candidates = [out_dir / "candidate_pairs.tsv", out_dir / "no_candidate_check.tsv"]
    for attempt, candidate in enumerate(candidates):
        command = [sys.executable, str(validator), "--matching", str(out_dir / "matching_results.tsv"),
                   "--candidate", str(candidate), "--test-dir", str(test_dir), *extra]
        result = subprocess.run(command, capture_output=True, text=True)
        record = {"command": command, "exit_code": result.returncode, "passed": result.returncode == 0,
                  "candidate_checked": attempt == 0, "stdout": result.stdout, "stderr_tail": result.stderr[-2000:]}
        if result.returncode == 0 or "FAIL" in result.stdout:
            break
    return record


def write(args) -> dict:
    import pyarrow as pa

    started = time.perf_counter()
    report = json.loads((args.out / "policy_report.json").read_text())
    cal = json.loads((args.out / "calibrator.json").read_text())
    if cal.get("run_id") != report["calibrator"].get("run_id"):
        raise ValueError(f"calibrator.json (run {cal.get('run_id')}) and policy_report.json (run "
                         f"{report['calibrator'].get('run_id')}) come from different tune runs; rerun --stage tune")
    spec = report["chosen"]
    log(f"write: policy {spec['name']} {spec['params']}")
    summary = write_outputs(spec, cal, args)
    pa.default_memory_pool().release_unused()
    summary["validator"] = run_validator(args.validator, args.out / "output", args.test_dir)
    summary["validation_predicted_nonempty_rate"] = spec["validation"]["predicted_nonempty_rate"]
    summary["seconds"] = time.perf_counter() - started
    report["test"] = summary
    atomic_write_json(args.out / "policy_report.json", report)
    write_markdown(args.out, report)
    log(f"write: {summary['matched_pairs']:,} matches over {summary['s1_lines']:,} S1 "
        f"(non-empty {summary['predicted_nonempty_rate']:.3f}; validation {spec['validation']['predicted_nonempty_rate']:.3f}); "
        f"one-owner conflicts {summary['one_owner']['targets_in_conflict']:,} (applied {spec['one_owner']}); "
        f"validator exit {summary['validator']['exit_code']}")
    print(summary["validator"]["stdout"][-1500:], flush=True)
    return summary


# ---------------------------------------------------------------------------
# Report


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.5f}"


def params_text(params: dict) -> str:
    return ", ".join(f"{k}={v:.4g}" for k, v in params.items())


def write_markdown(out: Path, report: dict) -> None:
    chosen, boot, sel = report["chosen"], report["holdout_bootstrap_chosen_minus_p1"], report["selection"]
    lines = [
        "# CE decision policy", "",
        f"Chosen: **{chosen['name']}** ({chosen['source']}, {params_text(chosen['params'])}, one-owner "
        f"{'yes' if chosen['one_owner'] else 'no'}): validation macro F0.5 {fmt(chosen['validation']['macro_f05'])}, "
        f"holdout {fmt(chosen['holdout']['macro_f05'])}.",
        f"Holdout chosen − P1: {boot['mean_diff']:+.5f} (paired S1 bootstrap 95% CI [{boot['ci95'][0]:+.5f}, "
        f"{boot['ci95'][1]:+.5f}], {boot['reps']} reps).", "",
        f"Oracle for these candidate lists (predict exactly the true pairs): validation "
        f"{fmt(report['oracle_macro_f05']['validation'])}, holdout {fmt(report['oracle_macro_f05']['holdout'])}. "
        f"Pair AUC of the logit: validation {fmt(report['pair_auc']['validation'])}, holdout "
        f"{fmt(report['pair_auc']['holdout'])}.", "",
        "| Policy | Params | Validation F0.5 | Holdout F0.5 | Val precision / recall | Val singleton acc | Eligible |",
        "|---|---|---:|---:|---|---:|---|"]
    for p in report["policies"]:
        v = p["validation"]
        lines.append(f"| {p['name']} | {params_text(p['params'])} | {fmt(v['macro_f05'])} | {fmt(p['holdout']['macro_f05'])} | "
                     f"{fmt(v['mean_precision'])} / {fmt(v['mean_recall'])} | {fmt(v['singleton_accuracy'])} | "
                     f"{'yes' if p['eligible'] else 'no'} |")
    lines += ["", f"Selection: {sel['rule']} (tolerance {sel['tie_tolerance']}). Stacked best on validation CV "
                  f"{fmt(sel['best_stacked_validation_cv'])} vs plain {fmt(sel['best_plain_validation'])}: "
                  f"{'included' if sel['stacked_included'] else 'not included'}. Stacked rows use out-of-fold scores "
                  "on validation.", "",
              "One-owner (validation only resolves conflicts inside the split; test resolves them across all test S1):", ""]
    for name, o in report["one_owner"].items():
        c = o["validation_conflicts"]
        lines.append(f"- {name}: validation delta {o['validation_delta']:+.5f}, holdout delta {o['holdout_delta']:+.5f}; "
                     f"validation claims dropped {c['claims_dropped']:,} ({c['dropped_false']:,} false, "
                     f"{c['dropped_true']:,} true)")
    test = report.get("test")
    if test:
        v = test["validator"]
        lines += ["", f"Test: {test['s1_lines']:,} S1, {test['candidate_pairs']:,} candidate pairs, "
                      f"{test['matched_pairs']:,} matches, non-empty rate {test['predicted_nonempty_rate']:.4f} "
                      f"(validation {test['validation_predicted_nonempty_rate']:.4f}); one-owner conflicts "
                      f"{test['one_owner']['targets_in_conflict']:,} targets / {test['one_owner']['claims_dropped']:,} "
                      f"claims (applied: {test['one_owner']['applied']}).",
                  f"Validator: exit {v['exit_code']} ({'PASS' if v['passed'] else 'FAIL'}; candidate cross-check "
                  f"{'on' if v['candidate_checked'] else 'off'})."]
    temporary = out / f"POLICY_REPORT.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, out / "POLICY_REPORT.md")


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True, help="<ce_out>/scores holding <split>.npy logits")
    parser.add_argument("--test-dir", type=Path, help="test TSV directory (write stage)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("tune", "write", "all"), required=True)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="for per-country slices (skipped if absent)")
    parser.add_argument("--validator", type=Path, default=VALIDATOR)
    parser.add_argument("--tie-tolerance", type=float, default=1e-5)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args(argv)
    if args.stage in ("write", "all") and args.test_dir is None:
        parser.error("--test-dir is required for the write stage")
    if args.stage in ("write", "all") and not args.validator.is_file():
        parser.error(f"validator not found: {args.validator} (pass --validator)")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("tune", "all"):
        tune(args)
    if args.stage in ("write", "all"):
        summary = write(args)
        if not summary["validator"]["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
