"""Phase 2A baselines on the 75k benchmark pairs, evaluated with official entity-level macro F0.5.

Run from code/business_entity_resolution, in the required order:

    python3 -m src.phase2a_train --config ../../configs/phase2a_benchmark.json --model rules
    python3 -m src.phase2a_train --config ../../configs/phase2a_benchmark.json --model logistic
    python3 -m src.phase2a_train --config ../../configs/phase2a_benchmark.json --model lightgbm

Training uses sampled train-split pairs. Thresholds are swept on validation
only; the holdout split (never used for fitting or tuning) is reported as-is.
Validation and holdout are scored chunk by chunk on every candidate pair.
"""

from __future__ import annotations

from . import phase2a_env  # noqa: F401  (thread limits before numpy)

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .evaluate_blocking import ROOT
from .evaluate_phase1c import atomic_write_json, load_truth, ordered_fold_ids, sha256_file
from .phase2a_env import ResourceGuard, ResourceStop, log
from .phase2a_eval import apply_policy, entity_fbeta, plateau, sweep
from .phase2a_pairs import FEATURES, Layout, TargetStore, sample_train_rows, split_of

F = {name: i for i, name in enumerate(FEATURES)}
NAN_FEATURES = tuple(name for name in FEATURES if name.startswith(("score_", "rank_", "name_", "addr_", "digits_"))
                     and name not in ("name_exact", "name_suffix_exact"))
RULES = {
    "R1_exact_name": lambda X: X[:, F["name_exact"]] == 1,
    "R2_exact_name_and_address_agrees_or_missing": lambda X: (X[:, F["name_exact"]] == 1) & (
        np.nan_to_num(X[:, F["addr_token_set"]], nan=1.0) >= 0.8),
    "R3_suffix_exact_and_address_agrees_or_missing": lambda X: (X[:, F["name_suffix_exact"]] == 1) & (
        np.nan_to_num(X[:, F["addr_token_set"]], nan=1.0) >= 0.8),
    "R4_strong_name_and_strong_address": lambda X: (np.nan_to_num(X[:, F["name_token_set"]]) >= 0.9) & (
        np.nan_to_num(X[:, F["addr_token_set"]]) >= 0.85),
    "R5_R2_or_R4": lambda X: RULES["R2_exact_name_and_address_agrees_or_missing"](X) | RULES["R4_strong_name_and_strong_address"](X),
}


def chunks_for(layout: Layout, config: dict, split: str) -> list[int]:
    result = []
    for path in sorted(layout.features.glob("chunk*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["split"] == split:
            result.append(meta["chunk"])
    low, high = config["split"][split]
    expected = (high - low) // config["chunk_s1"]
    if len(result) != expected:
        raise RuntimeError(f"{split}: {len(result)}/{expected} feature chunks present; finish the features stage first")
    return result


def load_training(layout: Layout, config: dict, guard: ResourceGuard) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for index in chunks_for(layout, config, "train"):
        guard.check(f"load train chunk {index}")
        with np.load(layout.feature_chunk(index), allow_pickle=False) as data:
            Xs.append(data["X"]); ys.append(data["label"])
    return np.concatenate(Xs), np.concatenate(ys)


def load_validation_sample(layout: Layout, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Train-style sample of validation pairs, used only for LightGBM early stopping."""
    s = config["sampling"]
    Xs, ys = [], []
    for index in chunks_for(layout, config, "validation"):
        with np.load(layout.pair_chunk(index), allow_pickle=False) as data:
            pairs = {name: data[name] for name in ("s1_pos", "label", "rrf_rank")}
        rows = sample_train_rows(pairs, s["seed"], index, s["hard_negatives_per_s1"], s["random_negatives_per_s1"])
        with np.load(layout.feature_chunk(index), allow_pickle=False) as data:
            Xs.append(data["X"][rows]); ys.append(data["label"][rows])
    return np.concatenate(Xs), np.concatenate(ys)


class Imputer:
    """NaN -> 0 plus explicit missing indicators, then standardization (fit on train only)."""

    def fit(self, X: np.ndarray) -> "Imputer":
        self.nan_cols = [F[name] for name in NAN_FEATURES]
        Z = self._expand(X)
        self.mean = Z.mean(axis=0)
        self.std = Z.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def _expand(self, X: np.ndarray) -> np.ndarray:
        indicators = np.isnan(X[:, self.nan_cols]).astype(np.float32)
        return np.hstack([np.nan_to_num(X, nan=0.0), indicators]).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((self._expand(X) - self.mean) / self.std).astype(np.float32)

    def names(self) -> list[str]:
        return list(FEATURES) + [f"isnan_{name}" for name in NAN_FEATURES]


def score_split(layout: Layout, config: dict, split: str, scorer, guard: ResourceGuard) -> dict:
    """Stream every candidate pair of a split through ``scorer`` (X -> score)."""
    scores, labels, s1_pos, chunk_ids, rows = [], [], [], [], []
    for index in chunks_for(layout, config, split):
        guard.check(f"score {split} chunk {index}")
        with np.load(layout.feature_chunk(index), allow_pickle=False) as data:
            X = data["X"]
            scores.append(np.asarray(scorer(X), dtype=np.float64))
            labels.append(data["label"]); s1_pos.append(data["s1_pos"])
            chunk_ids.append(np.full(len(X), index, np.int16)); rows.append(data["rows"])
    return {"score": np.concatenate(scores), "label": np.concatenate(labels), "s1_pos": np.concatenate(s1_pos),
            "chunk": np.concatenate(chunk_ids), "row": np.concatenate(rows)}


def entity_table(layout: Layout, config: dict, split: str, context: dict) -> dict:
    """Per-S1 metadata for a split: truth size (incl. blocking misses), slices."""
    low, high = config["split"][split]
    positions = np.arange(low, high)
    s1 = context["s1"]
    truth_indptr, truth_values = context["truth"]
    missing_links = np.zeros(len(positions), np.int32)
    for i, p in enumerate(positions):
        truth = truth_values[truth_indptr[p]:truth_indptr[p + 1]]
        missing_links[i] = sum(context["target_missing"].get(int(t), False) for t in truth)
    return {"positions": positions, "truth_len": np.diff(truth_indptr)[positions],
            "country": s1["country"][positions], "s1_address_missing": s1["address_missing"][positions],
            "missing_address_truth_links": missing_links}


def evaluate(scored: dict, entities: dict, predicted: np.ndarray) -> dict:
    group = scored["s1_pos"] - entities["positions"][0]
    n = len(entities["positions"])
    f = entity_fbeta(group, scored["label"], predicted, entities["truth_len"])
    n_pred = np.bincount(group, weights=predicted, minlength=n)
    tp = np.bincount(group, weights=predicted & scored["label"], minlength=n)
    retrieved = np.bincount(group, weights=scored["label"], minlength=n)
    truth_len = entities["truth_len"]
    singleton = truth_len == 0
    slices = {"singleton": singleton, "positive": ~singleton,
              "s1_address_missing": entities["s1_address_missing"],
              "has_missing_address_truth_link": entities["missing_address_truth_links"] > 0}
    for country in sorted(set(entities["country"].tolist())):
        slices[f"country={country}"] = entities["country"] == country
    return {
        "macro_f05": float(f.mean()),
        "entities": int(n),
        "predicted_match_rate": float(np.mean(n_pred > 0)),
        "mean_predicted_set_size": float(n_pred.mean()),
        "pair_precision": float(tp.sum() / n_pred.sum()) if n_pred.sum() else None,
        "link_recall_vs_all_gold": float(tp.sum() / truth_len.sum()),
        "link_recall_vs_retrieved_gold": float(tp.sum() / retrieved.sum()),
        "singleton_accuracy": float(np.mean(n_pred[singleton] == 0)) if singleton.any() else None,
        "slices_macro_f05": {name: {"entities": int(mask.sum()), "macro_f05": float(f[mask].mean()) if mask.any() else None}
                             for name, mask in slices.items()},
        "errors": {
            "false_positive_pairs": int((n_pred - tp).sum()),
            "false_negative_links_matcher": int((retrieved - tp).sum()),
            "false_negative_links_blocking": int((truth_len - retrieved).sum()),
            "singletons_with_false_merge": int(np.sum(singleton & (n_pred > 0))),
        },
    }


def oracle(scored: dict, entities: dict) -> dict:
    """Perfect matcher on the frozen candidates: the blocker-limited ceiling."""
    return evaluate(scored, entities, scored["label"].copy())


def examples(scored: dict, predicted: np.ndarray, layout: Layout, context: dict, limit: int = 8) -> dict:
    """Deterministic false-positive / matcher false-negative examples (training data only)."""
    store = context["store"]
    s1 = context["s1"]
    out = {}
    for kind, mask in (("false_positive", predicted & ~scored["label"]), ("false_negative", ~predicted & scored["label"])):
        idx = np.flatnonzero(mask)
        if len(idx) > limit:
            idx = idx[np.linspace(0, len(idx) - 1, limit).astype(int)]
        rows = []
        cache = {}
        for i in idx:
            chunk = int(scored["chunk"][i])
            if chunk not in cache:
                with np.load(layout.pair_chunk(chunk), allow_pickle=False) as data:
                    cache[chunk] = data["cand"]
            cand = int(cache[chunk][scored["row"][i]])
            trow = int(store.rows(np.asarray([cand], np.uint32))[0])
            p = int(scored["s1_pos"][i])
            rows.append({"s1": str(s1["entity_id"][p]), "s1_name": str(s1["name"][p]), "s1_address": str(s1["address"][p]),
                         "candidate": f"S{3 if cand & 1 else 2}-{cand >> 1}", "cand_name": store.text(trow, "name"),
                         "cand_address": store.text(trow, "address"), "score": round(float(scored["score"][i]), 4)})
        out[kind] = rows
    return out


def thresholds(config: dict) -> tuple[np.ndarray, np.ndarray]:
    grid = config["thresholds"]
    return np.linspace(0.02, 0.98, grid["grid"]), np.linspace(0.3, 0.95, grid["empty_guard_grid"])


def fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def run_model(model: str, config: dict, layout: Layout, guard: ResourceGuard, context: dict) -> dict:
    out = layout.out / "runs" / model
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    entities = {split: entity_table(layout, config, split, context) for split in ("validation", "holdout")}
    result = {"model": model, "features": list(FEATURES)}
    if model == "rules":
        per_rule = {}
        for name, rule in RULES.items():
            scored = {split: score_split(layout, config, split, lambda X, r=rule: r(X).astype(np.float64), guard)
                      for split in ("validation", "holdout")}
            per_rule[name] = {split: evaluate(scored[split], entities[split], scored[split]["score"] >= 0.5)
                              for split in scored}
        best = max(per_rule, key=lambda name: per_rule[name]["validation"]["macro_f05"])
        scored = {split: score_split(layout, config, split, lambda X: RULES[best](X).astype(np.float64), guard)
                  for split in ("validation", "holdout")}
        result.update(rules=per_rule, selected=best, selection="highest validation macro F0.5",
                      validation=per_rule[best]["validation"], holdout=per_rule[best]["holdout"],
                      empty_prediction_baseline={s: evaluate(scored[s], entities[s], np.zeros(len(scored[s]["score"]), bool))
                                                 for s in scored},
                      oracle_ceiling={s: oracle(scored[s], entities[s]) for s in scored},
                      holdout_examples=examples(scored["holdout"], scored["holdout"]["score"] >= 0.5, layout, context))
    else:
        X, y = load_training(layout, config, guard)
        result["train_rows"] = int(len(y)); result["train_positives"] = int(y.sum())
        if model == "logistic":
            from sklearn.linear_model import LogisticRegression
            imputer = Imputer().fit(X)
            Z = imputer.transform(X)
            del X
            X = None
            candidates = {}
            for C in config["models"]["logistic"]["C"]:
                guard.check(f"logistic C={C}")
                fits = []
                for repeat in range(2):          # reproducibility: two identical fits
                    clf = LogisticRegression(C=C, max_iter=config["models"]["logistic"]["max_iter"],
                                             class_weight=config["models"]["logistic"]["class_weight"])
                    clf.fit(Z, y)
                    fits.append(clf)
                identical = np.array_equal(fits[0].coef_, fits[1].coef_) and np.array_equal(fits[0].intercept_, fits[1].intercept_)
                clf = fits[0]
                scorer = lambda Xc, c=clf: c.predict_proba(imputer.transform(Xc))[:, 1]
                val = score_split(layout, config, "validation", scorer, guard)
                t, e = thresholds(config)
                sw = sweep(val["s1_pos"] - entities["validation"]["positions"][0], val["label"], val["score"],
                           entities["validation"]["truth_len"], t, e)
                candidates[C] = {"clf": clf, "sweep": sw, "reproducible": identical, "val": val}
                log(f"logistic C={C}: validation macro F0.5 {sw['best']['macro_f05']:.4f} reproducible={identical}")
            C = max(candidates, key=lambda c: candidates[c]["sweep"]["best"]["macro_f05"])
            chosen = candidates[C]
            model_obj = {"imputer": imputer, "clf": chosen["clf"]}
            scorer = lambda Xc: chosen["clf"].predict_proba(imputer.transform(Xc))[:, 1]
            result["hyperparameters"] = {"C": C, "all_C": {str(c): v["sweep"]["best"] for c, v in candidates.items()}}
            result["reproducible_refit_identical"] = all(v["reproducible"] for v in candidates.values())
            result["coefficients"] = dict(zip(imputer.names(), map(float, chosen["clf"].coef_[0])))
            val = chosen["val"]
            best = chosen["sweep"]
        else:
            import lightgbm as lgb
            params = dict(config["models"]["lightgbm"])
            rounds, stopping = params.pop("num_boost_round"), params.pop("early_stopping_rounds")
            Xv, yv = load_validation_sample(layout, config)
            guard.check("lightgbm train")
            booster = lgb.train(params, lgb.Dataset(X, y, feature_name=list(FEATURES), free_raw_data=True),
                                num_boost_round=rounds, valid_sets=[lgb.Dataset(Xv, yv, feature_name=list(FEATURES))],
                                callbacks=[lgb.early_stopping(stopping, verbose=False)])
            del Xv, yv
            model_obj = booster
            scorer = lambda Xc: booster.predict(Xc, num_iteration=booster.best_iteration)
            val = score_split(layout, config, "validation", scorer, guard)
            t, e = thresholds(config)
            best = sweep(val["s1_pos"] - entities["validation"]["positions"][0], val["label"], val["score"],
                         entities["validation"]["truth_len"], t, e)
            result["hyperparameters"] = {**config["models"]["lightgbm"], "best_iteration": booster.best_iteration}
            result["feature_importance_gain"] = dict(sorted(zip(FEATURES, map(float, booster.feature_importance("gain"))),
                                                            key=lambda kv: -kv[1]))
        del y
        hold = score_split(layout, config, "holdout", scorer, guard)
        policy = best["best"]
        predict = lambda s, ent: apply_policy(s["s1_pos"] - ent["positions"][0], s["score"], len(ent["positions"]),
                                              policy["threshold"], policy["empty_threshold"])
        no_guard = {"threshold": max((r for r in best["grid"] if r["empty_threshold"] is None),
                                     key=lambda r: r["macro_f05"])["threshold"]}
        from sklearn.metrics import average_precision_score, roc_auc_score
        result.update(
            policy=policy, policy_selection="grid sweep on validation only; holdout untouched",
            threshold_plateau_no_guard=plateau(best["grid"], max((r for r in best["grid"] if r["empty_threshold"] is None),
                                                                 key=lambda r: r["macro_f05"])),
            best_without_empty_guard=no_guard,
            validation=evaluate(val, entities["validation"], predict(val, entities["validation"])),
            holdout=evaluate(hold, entities["holdout"], predict(hold, entities["holdout"])),
            oracle_ceiling={"validation": oracle(val, entities["validation"]), "holdout": oracle(hold, entities["holdout"])},
            pair_diagnostics={"validation_roc_auc": float(roc_auc_score(val["label"], val["score"])),
                              "validation_average_precision": float(average_precision_score(val["label"], val["score"]))},
            holdout_score_fingerprint=fingerprint(hold["score"].astype(np.float32)),
            holdout_examples=examples(hold, predict(hold, entities["holdout"]), layout, context),
            threshold_grid=best["grid"],
        )
        with (out / "model.pkl").open("wb") as file:
            pickle.dump(model_obj, file)
    result["runtime_seconds"] = time.perf_counter() - started
    result["resources"] = guard.summary()
    atomic_write_json(out / "metrics.json", result)
    log(f"{model}: validation macro F0.5 {result['validation']['macro_f05']:.4f}, "
        f"holdout {result['holdout']['macro_f05']:.4f} (oracle holdout "
        f"{result['oracle_ceiling']['holdout']['macro_f05']:.4f})")
    return result


def build_context(config: dict, layout: Layout, root: Path = ROOT) -> dict:
    with np.load(layout.texts / "s1.npz", allow_pickle=False) as data:
        s1 = {name: data[name] for name in data.files}
    p1c = json.loads((root / config["inputs"]["phase1c_config"]).read_text())
    ordered = ordered_fold_ids(root / config["inputs"]["folds"], p1c["algorithm"]["validation_fold"], p1c["algorithm"]["seed"])
    ids = ordered[:len(s1["entity_id"])]
    if ids != s1["entity_id"].tolist():
        raise RuntimeError("S1 store order differs from the fold order")
    truth = load_truth(root / config["inputs"]["train_dir"], ids)
    target_missing = {}
    for source in (2, 3):
        with np.load(root / config["inputs"]["phase1c_work_dir"] / "meta" / f"S{source}_truth_targets.npz",
                     allow_pickle=False) as data:
            target_missing.update(zip(data["ids"].tolist(), data["address_missing"].tolist()))
    return {"s1": s1, "truth": truth, "target_missing": target_missing, "store": TargetStore(layout.texts)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("rules", "logistic", "lightgbm"), required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    layout = Layout(ROOT / config["paths"]["output_dir"])
    order = ("rules", "logistic", "lightgbm")
    for earlier in order[:order.index(args.model)]:
        if not (layout.out / "runs" / earlier / "metrics.json").exists():
            raise SystemExit(f"Run --model {earlier} first (required order: {' -> '.join(order)})")
    if args.model == "lightgbm":
        logistic = json.loads((layout.out / "runs/logistic/metrics.json").read_text())
        if not logistic.get("reproducible_refit_identical"):
            raise SystemExit("Logistic regression was not reproducible; LightGBM is gated on it")
    resources = config["resources"]
    guard = ResourceGuard(resources["min_available_memory_gib"], resources["max_swap_growth_gib"],
                          f"cd code/business_entity_resolution && python3 -m src.phase2a_train --config {args.config} --model {args.model}")
    try:
        context = build_context(config, layout)
        result = run_model(args.model, config, layout, guard, context)
        result["config_sha256"] = sha256_file(config_path)
        atomic_write_json(layout.out / "runs" / args.model / "metrics.json", result)
    except ResourceStop as stop:
        log(str(stop))
        raise SystemExit(3)


if __name__ == "__main__":
    main()
