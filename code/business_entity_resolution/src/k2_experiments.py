"""K2: full-fold stage-2 matcher experiments, champion, and error budget (official macro F0.5).

Run from code/business_entity_resolution (resumable; finished experiments are skipped):

    python3 -m src.k2_experiments --config ../../configs/k2_matcher.json \
        --phase1c-work-dir <Phase 1C work dir> --k1-dir <K1 output dir with models/> \
        --work-dir /tmp/k2 --output-dir /kaggle/working/k2 --stage all

Stages: ``stores`` (K1 token/text stores) -> ``features`` (filtered lists + stage-2
features, multi-process) -> ``experiments`` -> report.

Protocol: models train on sampled train-split pairs (all positives, the top filter-ranked
negatives, and seeded random negatives weighted back to the natural rate). Thresholds and
the empty-set guard are chosen on validation (every kept pair). Each S1's truth includes
gold dropped by the filter and missed by the blocker. The holdout is scored once, only
for the champion.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json
from .k1_filter import Stores, build_stores, load_context
from .k2_features import ALL_FEATURES, GROUPS, build_features, group_bounds
from .phase2a_analysis import lookup_owner, owner_map
from .phase2a_env import ResourceGuard, ResourceStop, log
from .phase2a_eval import apply_policy, entity_fbeta, group_max, plateau

FI = {name: i for i, name in enumerate(ALL_FEATURES)}
BUCKETS = ("0", "1", "2", "3", "4", "5", "6+")


# ---------------------------------------------------------------------------
# Data


def sample_rows(data: dict, seed: int, index: int, top_k: int, random_count: int) -> tuple[np.ndarray, np.ndarray]:
    """All positives, the top filter-ranked negatives, and random negatives weighted to the natural rate."""
    rng = np.random.default_rng([seed, index])
    label, frank = data["label"], data["X"][:, FI["filter_rank"]]
    _, starts, sizes = group_bounds(data["s1_pos"])
    rows, weights = [], []
    for start, size in zip(starts, sizes):
        idx = np.arange(start, start + size)
        positives = idx[label[idx]]
        negatives = idx[~label[idx]]
        negatives = negatives[np.argsort(frank[negatives], kind="stable")]
        top, rest = negatives[:top_k], negatives[top_k:]
        picked = rng.choice(rest, size=min(random_count, len(rest)), replace=False) if len(rest) else rest
        rows += [positives, top, picked]
        weights += [np.ones(len(positives)), np.ones(len(top)), np.full(len(picked), len(rest) / max(len(picked), 1))]
    return np.concatenate(rows).astype(np.int64), np.concatenate(weights).astype(np.float32)


def load_split(work: Path, config: dict, ctx: dict, split: str, *, sample: bool, extra: list | None = None) -> dict:
    """Pairs and per-S1 arrays for a split. ``extra`` adds whole additional training folds (train only).

    Extra-fold positions are offset by ``fold * 10,000,000`` so they never collide with fold-0 positions;
    each row carries its own country code from its fold's context.
    """
    low, high = config["split"][split]
    chunk = config["chunk_s1"]
    s = config["train_sampling"]
    sources = [(work / "features", range(low // chunk, (high + chunk - 1) // chunk), ctx, 0)]
    if split == "train" and extra:
        for e in extra:
            n_k = len(e["ctx"]["ordered"])
            sources.append((work / e["features_dir"], range(0, (n_k + chunk - 1) // chunk), e["ctx"], e["offset"]))
    parts = {name: [] for name in ("X", "label", "s1_pos", "cand", "weight", "country")}
    entities = {name: [] for name in ("ent_positions", "ent_truth_len", "ent_kept_truth", "ent_retrieved_truth",
                                      "ent_cand_count", "ent_country")}
    for directory, indices, fold_ctx, offset in sources:
        for index in indices:
            with np.load(directory / f"chunk{index:03d}.npz", allow_pickle=False) as data:
                data = {name: data[name] for name in data.files}
            if sample:
                rows, weight = sample_rows(data, s["seed"] + offset, index, s["top_filter_negatives"], s["random_negatives"])
            else:
                rows, weight = np.arange(len(data["label"])), np.ones(len(data["label"]), np.float32)
            local = data["s1_pos"][rows]
            parts["X"].append(data["X"][rows]); parts["label"].append(data["label"][rows])
            parts["s1_pos"].append(local + offset); parts["cand"].append(data["cand"][rows]); parts["weight"].append(weight)
            parts["country"].append(fold_ctx["country_code"][local])
            entities["ent_positions"].append(data["s1_positions"] + offset)
            entities["ent_truth_len"].append(data["s1_truth_len"]); entities["ent_kept_truth"].append(data["s1_kept_truth"])
            entities["ent_retrieved_truth"].append(data["s1_retrieved_truth"]); entities["ent_cand_count"].append(data["s1_cand_count"])
            entities["ent_country"].append(fold_ctx["country_code"][data["s1_positions"]])
    out = {name: np.concatenate(values) for name, values in parts.items()}
    out.update({name: np.concatenate(values) for name, values in entities.items()})
    out["group"] = np.searchsorted(out["ent_positions"], out["s1_pos"])
    return out


def subset_entities(data: dict, keep_entities: np.ndarray) -> dict:
    """Restrict a split to a boolean mask over its entities (pairs follow their S1)."""
    new_index = np.cumsum(keep_entities) - 1
    rows = keep_entities[data["group"]]
    out = {name: data[name][rows] for name in ("X", "label", "s1_pos", "cand", "weight", "country")}
    out.update({name: data[name][keep_entities] for name in data if name.startswith("ent_")})
    out["group"] = new_index[data["group"][rows]]
    return out


def entity_flags(ctx: dict, stores: Stores, positions: np.ndarray) -> dict:
    """Per-S1 slice flags: gold with a missing/non-ASCII target address, S1 address missing, bucket."""
    indptr, values = ctx["truth"]
    rows = stores.rows(values.astype(np.uint32)) if len(values) else np.zeros(0, np.int64)
    missing = np.asarray(stores.t["addr_missing"][rows]) if len(rows) else np.zeros(0, bool)
    non_ascii = np.asarray(stores.t["non_ascii"][rows]) if len(rows) else np.zeros(0, bool)
    owner = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    has_missing = np.bincount(owner[missing], minlength=len(indptr) - 1) > 0
    has_non_ascii = np.bincount(owner[non_ascii], minlength=len(indptr) - 1) > 0
    q = ctx["queries"]
    return {"has_missing_address_gold": has_missing[positions], "has_non_ascii_gold": has_non_ascii[positions],
            "s1_address_missing": np.asarray([not q["address"][p] for p in positions]),
            "bucket": np.asarray([BUCKETS[min(int(v), 6)] for v in np.diff(indptr)[positions]])}


# ---------------------------------------------------------------------------
# Models and evaluation


def feature_index(groups: list[str]) -> np.ndarray:
    names = [name for group in groups for name in GROUPS[group]]
    return np.asarray([FI[name] for name in names], dtype=np.int64)


def fit(params: dict, train: dict, valid: dict, idx: np.ndarray, rows: np.ndarray | None = None):
    import lightgbm as lgb

    p = dict(params)
    rounds, stop = p.pop("num_boost_round"), p.pop("early_stopping_rounds")
    names = [ALL_FEATURES[i] for i in idx]
    full = len(idx) == len(ALL_FEATURES)
    X = train["X"] if rows is None else train["X"][rows]
    X = X if full else X[:, idx]
    y = train["label"] if rows is None else train["label"][rows]
    w = train["weight"] if rows is None else train["weight"][rows]
    dtrain = lgb.Dataset(X, y, weight=w, feature_name=names, free_raw_data=True)
    dvalid = lgb.Dataset(valid["X"] if full else valid["X"][:, idx], valid["label"], reference=dtrain)
    started = time.perf_counter()
    booster = lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=[dvalid],
                        callbacks=[lgb.early_stopping(stop, verbose=False)])
    return booster, {"rows": int(len(y)), "positives": int(y.sum()), "best_iteration": int(booster.best_iteration),
                     "seconds": time.perf_counter() - started}


def predict(booster, data: dict, idx: np.ndarray, threads: int) -> np.ndarray:
    X = data["X"] if len(idx) == len(ALL_FEATURES) else data["X"][:, idx]
    return booster.predict(X, num_iteration=booster.best_iteration, num_threads=threads)


def f_from_counts(tp: np.ndarray, predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    fp, fn = predicted - tp, truth - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(tp > 0, 1.25 * tp / (1.25 * tp + fp + 0.25 * fn), 0.0)
    return np.where(truth == 0, (predicted == 0).astype(float), f)


def tune_policy(data: dict, score: np.ndarray, config: dict) -> dict:
    """Coarse grid then a fine grid around the best threshold; optional empty-set guard."""
    n = len(data["ent_positions"])
    best_score = group_max(data["group"], score, n)

    def macro(t, e=None):
        predicted = apply_policy(data["group"], score, n, t, e, best_score)
        return float(entity_fbeta(data["group"], data["label"], predicted, data["ent_truth_len"]).mean())

    grid = config["thresholds"]
    coarse = [(float(t), macro(t)) for t in np.linspace(0.02, 0.98, grid["grid"])]
    t0 = max(coarse, key=lambda r: (r[1], r[0]))[0]
    step = 0.96 / (grid["grid"] - 1)
    fine = [(float(t), macro(t)) for t in np.linspace(max(t0 - step, 0.001), min(t0 + step, 0.999), 41)]
    t_best, f_best = max(coarse + fine, key=lambda r: (r[1], r[0]))
    guard = [(float(e), macro(t_best, e)) for e in np.linspace(t_best, min(t_best + 0.3, 0.999), grid["empty_guard_grid"])[1:]]
    e_best, f_guard = max(guard, key=lambda r: (r[1], -r[0])) if guard else (None, -1)
    near = [t for t, f in coarse + fine if f >= f_best - 0.001]
    return {"threshold": t_best, "empty_threshold": e_best if f_guard > f_best + 1e-9 else None,
            "macro_f05": max(f_best, f_guard), "plateau_0.001": [min(near), max(near)]}


def evaluate(data: dict, score: np.ndarray, policy: dict, flags: dict, ctx: dict) -> dict:
    n = len(data["ent_positions"])
    predicted = apply_policy(data["group"], score, n, policy["threshold"], policy["empty_threshold"])
    f = entity_fbeta(data["group"], data["label"], predicted, data["ent_truth_len"])
    n_pred = np.bincount(data["group"], weights=predicted, minlength=n)
    tp = np.bincount(data["group"], weights=predicted & data["label"], minlength=n)
    truth = data["ent_truth_len"]
    singleton = truth == 0
    slices = {"singleton": singleton, "positive": ~singleton, **{f"bucket={b}": flags["bucket"] == b for b in BUCKETS},
              "has_missing_address_gold": flags["has_missing_address_gold"],
              "has_non_ascii_gold": flags["has_non_ascii_gold"], "s1_address_missing": flags["s1_address_missing"]}
    for code, key in enumerate(ctx["keys"]):
        slices[f"country={ctx['raw_by_key'][key]}"] = data["ent_country"] == code
    return {
        "macro_f05": float(f.mean()), "entities": int(n), "policy": policy,
        "predicted_match_rate": float(np.mean(n_pred > 0)), "mean_predicted_set_size": float(n_pred.mean()),
        "pair_precision": float(tp.sum() / n_pred.sum()) if n_pred.sum() else None,
        "recall_all_gold": float(tp.sum() / truth.sum()), "recall_kept_gold": float(tp.sum() / data["ent_kept_truth"].sum()),
        "predicted_pairs": int(n_pred.sum()), "true_positive_pairs": int(tp.sum()), "false_positive_pairs": int((n_pred - tp).sum()),
        "entity_precision_mean": float(np.mean(tp[n_pred > 0] / n_pred[n_pred > 0])) if (n_pred > 0).any() else None,
        "entity_recall_mean": float(np.mean(tp[truth > 0] / truth[truth > 0])) if (truth > 0).any() else None,
        "predicted_entities_with_any_false_positive": float(np.mean((n_pred - tp)[n_pred > 0] > 0)) if (n_pred > 0).any() else None,
        "singleton_accuracy": float(np.mean(n_pred[singleton] == 0)) if singleton.any() else None,
        "oracle_macro_f05_kept": float(f_from_counts(data["ent_kept_truth"], data["ent_kept_truth"], truth).mean()),
        "slices": {name: {"entities": int(mask.sum()), "macro_f05": float(f[mask].mean()) if mask.any() else None}
                   for name, mask in slices.items()},
        "_f": f, "_predicted": predicted,
    }


def paired_ci(fa: np.ndarray, fb: np.ndarray, resamples: int, seed: int) -> dict:
    d = fb - fa
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, len(d), size=(50, len(d)))].mean(axis=1) for _ in range(resamples // 50)])
    return {"delta": float(d.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Error budget and reassignment


def error_budget(data: dict, score: np.ndarray, predicted: np.ndarray, owners: tuple, ctx: dict, stores: Stores,
                 examples_per_category: int = 5) -> dict:
    X, label, group = data["X"], data["label"], data["group"]
    n = len(data["ent_positions"])
    truth, kept, retrieved = data["ent_truth_len"], data["ent_kept_truth"], data["ent_retrieved_truth"]
    tp = np.bincount(group, weights=predicted & label, minlength=n)
    n_pred = np.bincount(group, weights=predicted, minlength=n)
    base = f_from_counts(tp, n_pred, truth).mean()
    fp, fn = predicted & ~label, ~predicted & label
    same_address = (X[:, FI["digits_all_s1_in_tgt"]] == 1) & (np.nan_to_num(X[:, FI["addr_overlap_frac"]]) >= 0.8)
    name_idf = np.nan_to_num(X[:, FI["name_idf_frac"]])
    owner = lookup_owner(owners[0], owners[1], data["cand"])
    fp_cats = {"fp_same_address_owned_by_other_s1": fp & same_address & (owner >= 0),
               "fp_same_address_unowned": fp & same_address & (owner < 0),
               "fp_other_address_strong_name": fp & ~same_address & (name_idf >= 0.8)}
    fp_cats["fp_other"] = fp & ~np.any(np.stack(list(fp_cats.values())), axis=0)
    remaining = fn.copy()
    fn_cats = {}
    for name, mask in (("fn_name_script_mismatch", X[:, FI["name_script_mismatch"]] == 1),
                       ("fn_target_address_missing", X[:, FI["tgt_addr_missing"]] == 1),
                       ("fn_digit_one_edit_typo", X[:, FI["digits_one_edit"]] == 1),
                       ("fn_digit_component_dropped", (X[:, FI["tgt_digits_all_in_s1"]] == 1) & (X[:, FI["digits_all_s1_in_tgt"]] == 0)),
                       ("fn_digit_other_mismatch", (X[:, FI["digits_s1_n"]] > 0) & (X[:, FI["digits_all_s1_in_tgt"]] == 0)),
                       ("fn_weak_name", name_idf < 0.3)):
        fn_cats[name] = remaining & mask
        remaining &= ~mask
    fn_cats["fn_other"] = remaining
    rows = {}
    for name, mask in {**fp_cats, **fn_cats}.items():
        k = np.bincount(group[mask], minlength=n)
        after = f_from_counts(tp, n_pred - k, truth) if name.startswith("fp") else f_from_counts(tp + k, n_pred + k, truth)
        rows[name] = {"pairs": int(mask.sum()), "delta_macro_f05_if_fixed": float(after.mean() - base)}
    singleton_fp = np.bincount(group[fp], minlength=n) * (truth == 0)
    rows["singleton_false_merges"] = {"entities": int((singleton_fp > 0).sum()),
                                      "delta_macro_f05_if_fixed": float(f_from_counts(tp, n_pred - singleton_fp, truth).mean() - base)}
    rows["gold_dropped_by_filter"] = {"links": int((retrieved - kept).sum()),
                                      "delta_upper_bound": float(f_from_counts(tp + retrieved - kept, n_pred + retrieved - kept, truth).mean() - base)}
    rows["gold_missed_by_blocker"] = {"links": int((truth - retrieved).sum()),
                                      "delta_upper_bound": float(f_from_counts(tp + truth - retrieved, n_pred + truth - retrieved, truth).mean() - base)}
    q = ctx["queries"]
    examples = {}
    for name, mask in {**fp_cats, **fn_cats}.items():
        idx = np.flatnonzero(mask)
        idx = idx[np.linspace(0, len(idx) - 1, min(examples_per_category, len(idx))).astype(int)] if len(idx) else idx
        out = []
        for i in idx:
            p, trow = int(data["s1_pos"][i]), int(stores.rows(data["cand"][i:i + 1])[0])
            out.append({"s1": f"{q['name'][p]} | {q['address'][p]}",
                        "candidate": f"{stores.text(trow, 'name')} | {stores.text(trow, 'addr')}", "score": round(float(score[i]), 4)})
        examples[name] = out
    return {"base_macro_f05": float(base), "categories": rows, "examples": examples}


def reassignment(fold_scores: dict, threshold: float, ctx: dict, splits: dict, owners: tuple) -> dict:
    """One S1 per record: if several fold S1 claim a candidate, keep the highest-scoring claim."""
    cand = np.concatenate([v["cand"] for v in fold_scores.values()]).astype(np.int64)
    pos = np.concatenate([v["s1_pos"] for v in fold_scores.values()])
    score = np.concatenate([v["score"] for v in fold_scores.values()])
    claim = score >= threshold
    c_cand, c_pos, c_score = cand[claim], pos[claim], score[claim]
    order = np.lexsort((c_pos, -c_score, c_cand))
    first = np.r_[True, c_cand[order][1:] != c_cand[order][:-1]]
    winner = dict(zip(c_cand[order][first].tolist(), c_pos[order][first].tolist()))
    claimants = Counter(c_cand.tolist())
    contested = {c for c, k in claimants.items() if k > 1}
    owner_ids = lookup_owner(owners[0], owners[1], np.asarray(sorted(contested), dtype=np.uint32)) if contested else np.zeros(0)
    num_to_pos = {int(v): i for i, v in enumerate(ctx["s1_num"])}
    wins = total = 0
    claim_sets: dict[int, set] = {}
    for c, p in zip(c_cand.tolist(), c_pos.tolist()):
        if c in contested:
            claim_sets.setdefault(c, set()).add(p)
    for c, oid in zip(sorted(contested), owner_ids.tolist()):
        owner_pos = num_to_pos.get(int(oid)) if oid >= 0 else None
        if owner_pos is not None and owner_pos in claim_sets[c]:
            total += 1
            wins += winner[c] == owner_pos
    result = {"claimed_pairs": int(claim.sum()), "contested_candidates": len(contested),
              "contests_with_true_owner_claiming": total, "owner_wins": wins,
              "win_precision": wins / total if total else None, "splits": {}}
    for name, (data, predicted) in splits.items():
        keep = np.asarray([not (c in contested and winner[c] != p) for c, p in
                           zip(data["cand"].astype(np.int64).tolist(), data["s1_pos"].tolist())]) if contested else np.ones(len(predicted), bool)
        after = predicted & keep
        n = len(data["ent_positions"])
        f_before = entity_fbeta(data["group"], data["label"], predicted, data["ent_truth_len"])
        f_after = entity_fbeta(data["group"], data["label"], after, data["ent_truth_len"])
        result["splits"][name] = {"macro_f05_before": float(f_before.mean()), "macro_f05_after": float(f_after.mean()),
                                  "delta": float(f_after.mean() - f_before.mean()),
                                  "false_positives_removed": int((predicted & ~after & ~data["label"]).sum()),
                                  "true_positives_lost": int((predicted & ~after & data["label"]).sum()), "entities": int(n)}
    return result


# ---------------------------------------------------------------------------
# Orchestration


def calibrate(score_valid: np.ndarray, label_valid: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(score_valid, label_valid.astype(float))


def run_experiments(config: dict, ctx: dict, work: Path, output: Path, guard: ResourceGuard, extra: list | None = None) -> dict:
    """Ablations on a fixed train-S1 budget, then the champion, E4, learning curve, and transfer.

    Each result is written as soon as it exists; a rerun skips finished results.
    """
    results_dir = output / "experiments"
    results_dir.mkdir(parents=True, exist_ok=True)
    stores = Stores(work)
    exp = config["experiments"]
    threads = config["lightgbm_ablation"]["num_threads"]
    train_start = config["split"]["train"][0]
    log("experiments: loading splits")
    train = load_split(work, config, ctx, "train", sample=True, extra=extra)
    log(f"experiments: {len(np.unique(train['s1_pos'])):,} train S1 ({len(train['label']):,} sampled pairs), "
        f"extra folds {[e['fold'] for e in extra or []]}")
    valid = load_split(work, config, ctx, "validation", sample=False)
    guard.check("loaded splits")
    flags_v = entity_flags(ctx, stores, valid["ent_positions"])
    owners = owner_map(ctx["train_dir"])
    decision = config["decision"]
    runs, f_arrays, boosters = {}, {}, {}

    def rows_upto(n_s1: int, extra=None) -> np.ndarray:
        mask = train["s1_pos"] < train_start + n_s1
        return np.flatnonzero(mask if extra is None else mask & extra)

    def ablation(name: str, groups: list[str], n_s1: int):
        path = results_dir / f"{name}.json"
        if path.exists():
            runs[name] = json.loads(path.read_text())
            f_arrays[name] = np.load(results_dir / f"{name}_f.npy")
            return
        guard.check(f"experiment {name}")
        idx = feature_index(groups)
        booster, info = fit(config["lightgbm_ablation"], train, valid, idx, rows_upto(n_s1))
        score = predict(booster, valid, idx, threads)
        policy = tune_policy(valid, score, config)
        result = evaluate(valid, score, policy, flags_v, ctx)
        np.save(results_dir / f"{name}_f.npy", result["_f"])
        f_arrays[name], boosters[name] = result["_f"], (booster, idx, score)
        runs[name] = {"groups": groups, "train_s1": n_s1, "training": info, "validation": public(result)}
        atomic_write_json(path, runs[name])
        log(f"{name}: validation macro F0.5 {result['macro_f05']:.4f} (groups {groups}, {n_s1:,} train S1, "
            f"iter {info['best_iteration']}, {info['seconds'] / 60:.1f} min)")

    n_ab = exp["ablation_train_s1"]
    ablation("E3", ["v1", "norm", "context"], n_ab)
    ablation("E2", ["v1", "norm"], n_ab)
    ablation("E1", ["v1"], n_ab)
    ablation("E0", ["v1"], exp["E0_train_s1"])
    comparisons = {"scale_E1_vs_E0": ("E0", "E1"), "norm_E2_vs_E1": ("E1", "E2"), "context_E3_vs_E2": ("E2", "E3"),
                   "all_E3_vs_E0": ("E0", "E3")}
    deltas = {name: paired_ci(f_arrays[a], f_arrays[b], exp["bootstrap"], 7) for name, (a, b) in comparisons.items()}

    def slice_change(a: str, b: str) -> float:
        sa, sb = runs[a]["validation"]["slices"], runs[b]["validation"]["slices"]
        changes = [sb[k]["macro_f05"] - sa[k]["macro_f05"] for k in sa if sa[k]["macro_f05"] is not None
                   and sb[k]["macro_f05"] is not None and sa[k]["entities"] >= 500]
        return float(min(changes)) if changes else 0.0

    adopt = {}
    for group, (a, b) in (("norm", ("E1", "E2")), ("context", ("E2", "E3"))):
        d = deltas[f"{group}_{b}_vs_{a}"]
        worst = slice_change(a, b)
        adopt[group] = {"delta": d["delta"], "ci95": d["ci95"], "worst_slice_change": worst,
                        "adopted": d["delta"] >= decision["min_gain"] and d["ci95"][0] > 0 and worst >= -decision["max_slice_drop"]}
    champion_groups = ["v1"] + [g for g in ("norm", "context") if adopt[g]["adopted"]]
    reference = {("v1", "norm", "context"): "E3", ("v1", "norm"): "E2", ("v1",): "E1"}[tuple(champion_groups)]
    atomic_write_json(results_dir / "ablation_summary.json", {"deltas": deltas, "adoption": adopt,
                                                              "champion_groups": champion_groups, "reference_ablation": reference})
    idx = feature_index(champion_groups)

    def reference_model():
        if reference not in boosters:
            booster, _ = fit(config["lightgbm_ablation"], train, valid, idx, rows_upto(n_ab))
            boosters[reference] = (booster, idx, predict(booster, valid, idx, threads))
        return boosters[reference]

    # Champion: every train S1, slower learning rate, isotonic calibration; holdout opened once.
    champion_path = results_dir / "champion.json"
    models = output / "models"
    if not champion_path.exists():
        guard.check("champion")
        booster, info = fit(config["lightgbm_champion"], train, valid, idx)
        raw_v = predict(booster, valid, idx, threads)
        iso = calibrate(raw_v, valid["label"])
        cal_v = iso.predict(raw_v)
        policy = tune_policy(valid, cal_v, config)
        val_eval = evaluate(valid, cal_v, policy, flags_v, ctx)
        models.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(models / "champion.txt"), num_iteration=booster.best_iteration)
        atomic_write_json(models / "champion_calibrator.json", {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
        atomic_write_json(models / "champion_policy.json", {"features": [ALL_FEATURES[i] for i in idx],
                                                            "groups": champion_groups, **policy})
        budget = error_budget(valid, cal_v, val_eval["_predicted"], owners, ctx, stores)
        holdout = load_split(work, config, ctx, "holdout", sample=False)
        cal_h = iso.predict(predict(booster, holdout, idx, threads))
        hold_eval = evaluate(holdout, cal_h, policy, entity_flags(ctx, stores, holdout["ent_positions"]), ctx)
        np.save(results_dir / "champion_valid_scores.npy", cal_v.astype(np.float32))
        np.save(results_dir / "champion_holdout_scores.npy", cal_h.astype(np.float32))
        # Compact pair-level record of the champion's decisions, for later model diagnostics
        # (e.g. testing pretrained encoders on the exact pairs the champion gets wrong).
        diag_cols = ["name_script_mismatch", "s1_name_script", "tgt_name_script", "digits_all_s1_in_tgt",
                     "addr_overlap_frac", "name_idf_frac", "name_token_set", "tgt_addr_missing", "digits_one_edit"]
        for split_name, data, cal in (("validation", valid, cal_v), ("holdout", holdout, cal_h)):
            np.savez_compressed(results_dir / f"champion_pairs_{split_name}.npz", s1_pos=data["s1_pos"], cand=data["cand"],
                                label=data["label"], score=cal.astype(np.float32),
                                diag=data["X"][:, [FI[c] for c in diag_cols]].astype(np.float16),
                                diag_columns=np.asarray(diag_cols))
        atomic_write_json(champion_path, {"groups": champion_groups, "training": info, "validation": public(val_eval),
                                          "holdout": public(hold_eval), "error_budget_validation": budget})
        log(f"champion: validation {val_eval['macro_f05']:.4f}, holdout {hold_eval['macro_f05']:.4f} "
            f"(iter {info['best_iteration']}, {info['seconds'] / 60:.1f} min)")
        del holdout
    champion = json.loads(champion_path.read_text())
    policy = champion["validation"]["policy"]

    # E4: calibrated out-of-fold scores for train S1 + champion scores for validation/holdout,
    # then one S1 per record (highest calibrated claim wins).
    e4_path = results_dir / "E4_reassignment.json"
    if not e4_path.exists():
        guard.check("E4")
        from sklearn.isotonic import IsotonicRegression  # noqa: F401

        folds = config["oof_folds"]
        chunk = config["chunk_s1"]
        cal_v = np.load(results_dir / "champion_valid_scores.npy").astype(np.float64)
        cal_h = np.load(results_dir / "champion_holdout_scores.npy").astype(np.float64)
        holdout = load_split(work, config, ctx, "holdout", sample=False)
        fold_scores = {"validation": {"cand": valid["cand"], "s1_pos": valid["s1_pos"], "score": cal_v},
                       "holdout": {"cand": holdout["cand"], "s1_pos": holdout["s1_pos"], "score": cal_h}}
        parts = {"cand": [], "s1_pos": [], "score": []}
        for f in range(folds):
            other = train["s1_pos"] % folds != f
            rows = rows_upto(config["split"]["train"][1] - train_start, other)
            if exp["oof_train_s1_per_fold"]:
                chosen = np.unique(train["s1_pos"][rows])[:exp["oof_train_s1_per_fold"]]
                rows = rows[np.isin(train["s1_pos"][rows], chosen)]
            booster_f, _ = fit(config["lightgbm_ablation"], train, valid, idx, rows)
            iso_f = calibrate(predict(booster_f, valid, idx, threads), valid["label"])
            for index in range(config["split"]["train"][0] // chunk, config["split"]["train"][1] // chunk):
                with np.load(work / "features" / f"chunk{index:03d}.npz", allow_pickle=False) as data:
                    mask = data["s1_pos"] % folds == f
                    X = data["X"][mask]
                    raw = booster_f.predict(X if len(idx) == len(ALL_FEATURES) else X[:, idx],
                                            num_iteration=booster_f.best_iteration, num_threads=threads)
                    parts["score"].append(iso_f.predict(raw))
                    parts["cand"].append(data["cand"][mask]); parts["s1_pos"].append(data["s1_pos"][mask])
            log(f"E4: out-of-fold scores for train fold {f} done")
        fold_scores["train_oof"] = {k: np.concatenate(v) for k, v in parts.items()}
        pred_v = apply_policy(valid["group"], cal_v, len(valid["ent_positions"]), policy["threshold"], policy["empty_threshold"])
        pred_h = apply_policy(holdout["group"], cal_h, len(holdout["ent_positions"]), policy["threshold"], policy["empty_threshold"])
        e4 = reassignment(fold_scores, policy["threshold"], ctx, {"validation": (valid, pred_v), "holdout": (holdout, pred_h)}, owners)
        e4["policy"] = policy
        e4["adopted"] = bool(e4["win_precision"] is not None and e4["win_precision"] >= decision["reassign_min_precision"]
                             and e4["splits"]["validation"]["delta"] > 0)
        e4["note"] = ("Competition only among fold-0 S1 (about 20% of all S1); at test time every S1 competes, "
                      "so the effect should be larger in both directions.")
        atomic_write_json(e4_path, e4)
        log(f"E4: validation delta {e4['splits']['validation']['delta']:+.4f}, win precision {e4['win_precision']}")
        del holdout

    # Learning curve (identical ablation settings; champion feature groups).
    lc_path = results_dir / "learning_curve.json"
    if not lc_path.exists():
        curve = {}
        for n_s1 in exp["learning_curve"]:
            guard.check(f"learning curve {n_s1}")
            if n_s1 == n_ab:
                booster, _, score = reference_model()
            else:
                booster, _ = fit(config["lightgbm_ablation"], train, valid, idx, rows_upto(n_s1))
                score = predict(booster, valid, idx, threads)
            curve[str(n_s1)] = tune_policy(valid, score, config)["macro_f05"]
            log(f"learning curve: {n_s1:,} train S1 -> {curve[str(n_s1)]:.4f}")
        if extra:
            booster, _ = fit(config["lightgbm_ablation"], train, valid, idx)
            n_all = int(len(np.unique(train["s1_pos"])))
            curve[str(n_all)] = tune_policy(valid, predict(booster, valid, idx, threads), config)["macro_f05"]
            log(f"learning curve: all folds ({n_all:,} train S1) -> {curve[str(n_all)]:.4f}")
        sizes = sorted(curve, key=int)
        lc = {"curve": curve, "last_step_gain": curve[sizes[-1]] - curve[sizes[-2]],
              "block_more_folds": curve[sizes[-1]] - curve[sizes[-2]] >= decision["lc_more_data_gain"]}
        atomic_write_json(lc_path, lc)

    # Country transfer (France proxy): train on one country, threshold from that country.
    st_path = results_dir / "country_transfer.json"
    if not st_path.exists():
        st = {}
        _, _, ref_score = reference_model()
        ref_policy = tune_policy(valid, ref_score, config)
        country_of_row = train["country"]
        for src, dst in (("us", "india"), ("india", "us")):
            if src not in ctx["keys"] or dst not in ctx["keys"]:
                continue
            guard.check(f"transfer {src}->{dst}")
            s_code, d_code = ctx["keys"].index(src), ctx["keys"].index(dst)
            booster, _ = fit(config["lightgbm_ablation"], train, valid, idx, rows_upto(exp["transfer_train_s1"], country_of_row == s_code))
            score = predict(booster, valid, idx, threads)
            src_rows = valid["ent_country"][valid["group"]] == s_code
            dst_rows = valid["ent_country"][valid["group"]] == d_code
            src_valid = subset_entities(valid, valid["ent_country"] == s_code)
            dst_valid = subset_entities(valid, valid["ent_country"] == d_code)
            dst_flags = entity_flags(ctx, stores, dst_valid["ent_positions"])
            src_policy = tune_policy(src_valid, score[src_rows], config)
            dst_oracle = tune_policy(dst_valid, score[dst_rows], config)
            ref_dst = evaluate(dst_valid, ref_score[dst_rows], ref_policy, dst_flags, ctx)
            transfer = evaluate(dst_valid, score[dst_rows], src_policy, dst_flags, ctx)
            st[f"{src}_to_{dst}"] = {
                "in_distribution_macro_f05": ref_dst["macro_f05"],
                "transfer_macro_f05_threshold_from_source": transfer["macro_f05"],
                "transfer_macro_f05_threshold_tuned_on_target": dst_oracle["macro_f05"],
                "drop": ref_dst["macro_f05"] - transfer["macro_f05"],
                "source_threshold": src_policy["threshold"], "target_best_threshold": dst_oracle["threshold"]}
            log(f"transfer {src}->{dst}: {st[f'{src}_to_{dst}']}")
        st["alarm"] = any(v["drop"] > decision["transfer_drop_alarm"] for v in st.values() if isinstance(v, dict))
        atomic_write_json(st_path, st)
    return summarize(config, output)


def summarize(config: dict, output: Path) -> dict:
    d = output / "experiments"
    read = lambda name: json.loads((d / name).read_text()) if (d / name).exists() else None
    result = {"ablations": {name: read(f"{name}.json") for name in ("E0", "E1", "E2", "E3")},
              "ablation_summary": read("ablation_summary.json"), "champion": read("champion.json"),
              "E4_reassignment": read("E4_reassignment.json"), "learning_curve": read("learning_curve.json"),
              "country_transfer": read("country_transfer.json")}
    atomic_write_json(output / "k2_results.json", result)
    (output / "K2_REPORT.md").write_text(make_report(result))
    return result


def make_report(r: dict) -> str:
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    lines = ["# K2 — full-fold stage-2 matcher experiments (fold 0, filtered candidate lists)", "",
             "Validation macro F0.5 (official metric; truth includes blocker misses and filter drops). Thresholds tuned on validation; "
             "holdout scored once for the champion.", "",
             "| Experiment | Feature groups | Train S1 | Validation F0.5 | Pair precision | Recall (all gold) | Singleton acc. | Iterations |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, run in r["ablations"].items():
        if run:
            v = run["validation"]
            lines.append(f"| {name} | {'+'.join(run['groups'])} | {run['train_s1'] or 'all'} | {v['macro_f05']:.4f} | "
                         f"{fmt(v['pair_precision'])} | {fmt(v['recall_all_gold'])} | {fmt(v['singleton_accuracy'])} | {run['training']['best_iteration']} |")
    if r["ablation_summary"]:
        lines += ["", "| Comparison | ΔF0.5 | 95% CI |", "|---|---:|---:|"]
        for name, d in r["ablation_summary"]["deltas"].items():
            lines.append(f"| {name} | {d['delta']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
        lines += ["", "Adoption: " + ", ".join(f"{g}: {'ADOPT' if a['adopted'] else 'reject'} (worst slice change {a['worst_slice_change']:+.4f})"
                                              for g, a in r["ablation_summary"]["adoption"].items()), ""]
    c = r["champion"]
    if c:
        lines += ["## Champion", "", f"Groups {'+'.join(c['groups'])}; policy {c['validation']['policy']}.", "",
                  "| Split | Macro F0.5 | Oracle (kept gold) | Pair precision | Recall (all gold) | Predicted-match rate | Singleton acc. |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for split in ("validation", "holdout"):
            v = c[split]
            lines.append(f"| {split} | {v['macro_f05']:.4f} | {v['oracle_macro_f05_kept']:.4f} | {fmt(v['pair_precision'])} | "
                         f"{fmt(v['recall_all_gold'])} | {fmt(v['predicted_match_rate'])} | {fmt(v['singleton_accuracy'])} |")
        lines += ["", "Holdout slices: " + "; ".join(f"{k} {fmt(s['macro_f05'])} (n={s['entities']:,})"
                                                     for k, s in c["holdout"]["slices"].items() if s["entities"]) + ".", "",
                  "### Error budget (validation): macro F0.5 gained if a category were fixed", "",
                  "| Category | Count | ΔF0.5 |", "|---|---:|---:|"]
        for name, row in sorted(c["error_budget_validation"]["categories"].items(),
                                key=lambda kv: -kv[1].get("delta_macro_f05_if_fixed", kv[1].get("delta_upper_bound", 0))):
            count = row.get("pairs", row.get("entities", row.get("links")))
            delta = row.get("delta_macro_f05_if_fixed", row.get("delta_upper_bound"))
            lines.append(f"| {name} | {count:,} | {delta:+.4f} |")
    if r["E4_reassignment"]:
        e = r["E4_reassignment"]
        lines += ["", f"## E4 one-S1-per-record reassignment: adopted = {e['adopted']}", "",
                  f"Contested candidates {e['contested_candidates']:,}; true owner among claimants {e['contests_with_true_owner_claiming']:,}; "
                  f"owner wins {e['owner_wins']:,} (win precision {fmt(e['win_precision'])}).",
                  "Validation ΔF0.5 " + f"{e['splits']['validation']['delta']:+.4f}; holdout ΔF0.5 {e['splits']['holdout']['delta']:+.4f}. {e['note']}"]
    if r["learning_curve"]:
        lines += ["", f"## Learning curve: {r['learning_curve']['curve']} — block more folds: {r['learning_curve']['block_more_folds']}"]
    if r["country_transfer"]:
        lines += ["", "## Country transfer (France proxy)", ""]
        for name, v in r["country_transfer"].items():
            if isinstance(v, dict):
                lines.append(f"- {name}: in-distribution {v['in_distribution_macro_f05']:.4f}, transfer {v['transfer_macro_f05_threshold_from_source']:.4f} "
                             f"(target-tuned threshold {v['transfer_macro_f05_threshold_tuned_on_target']:.4f}); drop {v['drop']:+.4f}; "
                             f"thresholds {v['source_threshold']:.3f} vs {v['target_best_threshold']:.3f}")
        lines.append(f"- alarm: {r['country_transfer']['alarm']}")
    return "\n".join(lines) + "\n"


def choose_filter_k(k1: dict, rules: dict) -> tuple[int | None, dict]:
    """Smallest K (<= max_k) whose gold retention meets the target on validation, holdout and, when
    required, the US->India transfer check (the France proxy). Pre-registered K1 rule."""
    target, max_k = rules["retention_target"], rules["max_k"]
    val = k1["by_split"]["validation"]["all"]["curve"]
    hold = k1["by_split"]["holdout"]["all"]["curve"]
    transfer = k1.get("transfer_us_to_india_validation") if rules.get("require_transfer", True) else None
    table = {}
    chosen = None
    for k in sorted(int(key) for key in val):
        row = {"validation": val[str(k)]["retention"], "holdout": hold[str(k)]["retention"],
               "transfer": transfer[str(k)]["retention"] if transfer else None,
               "mean_kept": val[str(k)]["mean_kept_candidates"], "oracle_f05": val[str(k)]["oracle_macro_f05"]}
        row["passes"] = bool(k <= max_k and row["validation"] >= target and row["holdout"] >= target
                             and (row["transfer"] is None or row["transfer"] >= target))
        table[str(k)] = row
        if chosen is None and row["passes"]:
            chosen = k
    return chosen, table


def extra_fold_context(config: dict, fold: int, work_k: Path, root: Path = ROOT) -> tuple[dict, dict]:
    """Config and context for an extra training fold (every S1 of the fold is training data)."""
    import copy

    from .evaluate_phase1c import ordered_fold_ids

    cfg = copy.deepcopy(config)
    cfg["inputs"]["phase1c_config"] = f"configs/phase1c_fold{fold}.json"
    cfg["inputs"]["phase1c_manifest"] = f"artifacts/phase1c_fold{fold}/fold{fold}_manifest.json"
    p1c = json.loads((root / cfg["inputs"]["phase1c_config"]).read_text())
    n_k = len(ordered_fold_ids(root / cfg["inputs"]["folds"], fold, p1c["algorithm"]["seed"]))
    cfg["split"] = {"unit": config["split"]["unit"], "train": [0, n_k], "validation": [n_k, n_k], "holdout": [n_k, n_k]}
    ctx_k = load_context(cfg, root, work_k)
    ctx_k["train_dir"] = root / cfg["inputs"]["train_dir"]
    return cfg, ctx_k


def prepare_extra_folds(config: dict, args, keep_k: int, guard: ResourceGuard) -> list:
    from .k1_filter import build_s1_store

    extra = []
    for fold in config.get("extra_train_folds", []):
        work_k = args.extra_folds_dir / f"fold{fold}" / "work"
        cfg_k, ctx_k = extra_fold_context(config, fold, work_k)
        if args.stage in ("features", "all"):
            s1_dir = args.work_dir / f"stores_fold{fold}"
            if not (s1_dir / "s1_name.npy").exists():
                build_s1_store(cfg_k, s1_dir, ctx_k)
            build_features(cfg_k, ctx_k, work_k, args.work_dir, args.k1_dir, keep_k, config["workers"], s1_dir=s1_dir,
                           features_dir=f"features_fold{fold}", filter_all_final=True)
        extra.append({"fold": fold, "ctx": ctx_k, "features_dir": f"features_fold{fold}", "offset": fold * 10_000_000})
        log(f"extra training fold {fold}: {len(ctx_k['ordered']):,} S1 ready")
    return extra


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase1c-work-dir", type=Path, required=True)
    parser.add_argument("--k1-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("stores", "features", "experiments", "all"), required=True)
    parser.add_argument("--extra-folds-dir", type=Path, default=None,
                        help="directory with fold<k>/work Phase 1C shards for the config's extra_train_folds")
    args = parser.parse_args(argv)
    config = json.loads(args.config.resolve().read_text())
    guard = ResourceGuard(1.0, 8.0, f"python3 -m src.k2_experiments ... --stage {args.stage}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ctx = load_context(config, ROOT, args.phase1c_work_dir)
        ctx["train_dir"] = ROOT / config["inputs"]["train_dir"]
        k1 = json.loads((args.k1_dir / "k1_results.json").read_text())
        chosen, gate_table = choose_filter_k(k1, config["filter"])
        keep_k = config["filter"]["k_override"] or chosen or config["filter"]["fallback_k"]
        source = "override" if config["filter"]["k_override"] else ("k1_curves_all_gates" if chosen else "fallback")
        atomic_write_json(args.output_dir / "filter_choice.json", {"keep_k": keep_k, "source": source,
                                                                  "gate_table": gate_table, "k1_decision": k1["decision"]})
        log(f"K2: keep top-{keep_k} per S1 (source: {source})")
        if args.stage in ("stores", "all"):
            build_stores(config, ROOT, args.work_dir, ctx, guard)
        if args.stage in ("features", "all"):
            build_features(config, ctx, args.phase1c_work_dir, args.work_dir, args.k1_dir, keep_k, config["workers"])
        extra = []
        if args.extra_folds_dir:
            extra = prepare_extra_folds(config, args, keep_k, guard)
        if args.stage in ("experiments", "all"):
            run_experiments(config, ctx, args.work_dir, args.output_dir, guard, extra)
    except ResourceStop as stop:
        log(str(stop))
        raise SystemExit(3)


if __name__ == "__main__":
    main()
