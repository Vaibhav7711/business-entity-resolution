import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.ce_policy import (NOTHING, apply_policy, best_threshold, calibrate, curve, evaluate, fit_calibrator,
                           load_split, main, one_owner, prepare_sweep, sigmoid)


def reference_f05(truth: set, pred: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(truth & pred)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def random_split(rng, n=80) -> tuple[dict, list[set]]:
    """An in-memory split (runs in shuffled S1 order) and each S1's full gold set, including gold not retrieved."""
    order = rng.permutation(n)
    g, label, t_code, truth_len, gold = [], [], [], np.zeros(n, np.int64), [set() for _ in range(n)]
    starts = [0]
    next_target = 0
    for s1 in order:
        rows = int(rng.integers(0, 7))
        for _ in range(rows):
            is_true = bool(rng.random() < 0.35)
            g.append(s1); label.append(is_true); t_code.append(next_target)
            if is_true:
                gold[s1].add(next_target)
            next_target += 1
        for _ in range(int(rng.random() < 0.25)):
            gold[s1].add(-1 - next_target)
            next_target += 1
        truth_len[s1] = len(gold[s1])
        if rows:
            starts.append(starts[-1] + rows)
    run_s1 = np.asarray([s1 for s1 in order if np.any(np.asarray(g) == s1)], np.int64)
    rows = len(g)
    d = {"n": n, "g": np.asarray(g, np.int32), "label": np.asarray(label, bool), "truth_len": truth_len,
         "starts": np.asarray(starts, np.int64), "run_s1": run_s1, "t_code": np.asarray(t_code, np.int32),
         "logit": rng.normal(size=rows).astype(np.float32), "filter_rank": np.zeros(rows, np.int16),
         "filter_score": rng.random(rows).astype(np.float32), "s1_ids": [f"S1-{i}" for i in range(n)]}
    return d, gold


def reference_macro(d: dict, gold: list[set], pred: np.ndarray) -> np.ndarray:
    chosen = [set() for _ in range(d["n"])]
    for row in np.flatnonzero(pred):
        chosen[d["g"][row]].add(int(d["t_code"][row]))
    return np.asarray([reference_f05(gold[i], chosen[i]) for i in range(d["n"])])


def test_metric_matches_pure_python_reference():
    rng = np.random.default_rng(0)
    for _ in range(20):
        d, gold = random_split(rng)
        pred = rng.random(len(d["g"])) < 0.5
        metrics, f = evaluate(d, pred)
        expected = reference_macro(d, gold, pred)
        assert np.allclose(f, expected)
        assert metrics["macro_f05"] == pytest.approx(expected.mean())


def test_singletons_empty_lists_and_missed_gold():
    # S1 0: singleton, nothing predicted -> 1. S1 1: singleton, one prediction -> 0.
    # S1 2: no candidates, truth 2 -> 0. S1 3: one of two gold retrieved and predicted -> P=1, R=0.5.
    # S1 4: truth 1, only a false prediction -> 0. S1 5: singleton without candidates -> 1.
    d = {"n": 6, "g": np.asarray([0, 1, 3, 4], np.int32), "label": np.asarray([False, False, True, False]),
         "truth_len": np.asarray([0, 0, 2, 2, 1, 0])}
    pred = np.asarray([False, True, True, True])
    _, f = evaluate(d, pred)
    assert f.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.25 * 0.5 / (0.25 + 0.5), 0.0, 1.0])
    metrics, f = evaluate(d, np.zeros(4, bool))
    assert f.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert metrics["singleton_accuracy"] == 1.0 and metrics["mean_precision"] is None


def test_sweeps_equal_direct_evaluation():
    rng = np.random.default_rng(1)
    for _ in range(10):
        d, _ = random_split(rng)
        score = np.round(rng.random(len(d["g"])), 1)          # many ties
        prep = prepare_sweep(d["g"], score, d["label"], d["truth_len"])
        macro = curve(prep)
        assert prep["thresholds"][0] == NOTHING
        for t, value in zip(prep["thresholds"], macro):
            assert value == pytest.approx(evaluate(d, score >= t)[0]["macro_f05"])
        s1_max = np.full(d["n"], -np.inf)
        np.maximum.at(s1_max, d["g"], score)
        for guard in (0.3, 0.7):
            t, value = best_threshold(prep, s1_max[prep["g"]] >= guard, cap=guard)
            spec = {"family": "empty_guard", "params": {"t": t, "t_empty": guard}, "one_owner": False}
            assert t == NOTHING or t <= guard
            assert value == pytest.approx(evaluate(d, apply_policy(spec, d, score)[0])[0]["macro_f05"])
        for r in (0.5, 0.9):
            spec = {"family": "relative", "params": {"t": 0.0, "r": r}, "one_owner": False}
            keep = apply_policy(spec, d, score)[0]
            t, value = best_threshold(prepare_sweep(d["g"][keep], score[keep], d["label"][keep], d["truth_len"]))
            spec["params"]["t"] = t
            assert value == pytest.approx(evaluate(d, apply_policy(spec, d, score)[0])[0]["macro_f05"])


def test_one_owner_keeps_best_claim_with_tie_breaks():
    # rows: S1 0: A .8, B .7 (logit 1, rank 3), D .9 | S1 1: A .9, C .6 (logit 1, rank 4) | S1 2: B .7 (logit 2, rank 5), C .6 (logit 1, rank 1)
    t = {"A": 0, "B": 1, "C": 2, "D": 3}
    d = {"g": np.asarray([0, 0, 0, 1, 1, 2, 2], np.int32),
         "t_code": np.asarray([t["A"], t["B"], t["D"], t["A"], t["C"], t["B"], t["C"]], np.int32),
         "logit": np.asarray([0, 1, 0, 0, 1, 2, 1], np.float32),
         "filter_rank": np.asarray([0, 3, 1, 0, 4, 5, 1], np.int16),
         "label": np.asarray([False, False, True, True, False, True, True]),
         "starts": np.asarray([0, 3, 5, 7])}
    score = np.asarray([0.8, 0.7, 0.9, 0.9, 0.6, 0.7, 0.6])
    kept, stats = one_owner(np.ones(7, bool), d, score)
    assert kept.tolist() == [False, False, True, True, False, True, True]
    assert stats == {"claims": 7, "targets_in_conflict": 3, "claims_dropped": 3, "dropped_true": 0, "dropped_false": 3}
    kept, stats = one_owner(score >= 0.75, d, score)             # A (S1 0 vs 1) is the only conflict left
    assert kept.tolist() == [False, False, True, True, False, False, False] and stats["claims_dropped"] == 1
    guard = {"family": "empty_guard", "params": {"t": 0.65, "t_empty": 0.85}, "one_owner": False}
    assert apply_policy(guard, d, score)[0].tolist() == [True, True, True, True, False, False, False]
    relative = {"family": "relative", "params": {"t": 0.0, "r": 0.8}, "one_owner": True}
    assert apply_policy(relative, d, score)[0].tolist() == [False, False, True, True, False, True, True]


def test_calibrator_matches_sklearn_isotonic():
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(2)
    logit = rng.normal(size=3000).astype(np.float32)
    label = rng.random(3000) < sigmoid(2 * logit)
    cal = fit_calibrator(logit, label)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(sigmoid(logit), label)
    probe = np.r_[logit, -9.0, 9.0].astype(np.float32)
    assert np.allclose(calibrate(cal, probe), iso.predict(sigmoid(probe)))


# ---------------------------------------------------------------------------
# Synthetic pairs roots, scores, and TSVs


def write_pairs(split_dir: Path, groups: list[dict], s1_order: list[str], labelled: bool, part_rows: int = 23) -> np.ndarray:
    """Parts cut every ``part_rows`` rows (S1 runs cross part boundaries); returns logits in row order."""
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ("s1_id", "t_id", "label", "filter_score", "filter_rank")}
    logits = []
    for group in groups:
        for rank, (t_id, label, logit) in enumerate(group["rows"]):
            rows["s1_id"].append(group["s1"]); rows["t_id"].append(t_id)
            rows["label"].append(int(label) if labelled else -1)
            rows["filter_score"].append(1.0 - rank / 50); rows["filter_rank"].append(rank)
            logits.append(logit)
    types = {"s1_id": pa.string(), "t_id": pa.string(), "label": pa.int8(), "filter_score": pa.float32(),
             "filter_rank": pa.int16()}
    table = pa.table({k: pa.array(v, types[k]) for k, v in rows.items()})
    for part, start in enumerate(range(0, max(table.num_rows, 1), part_rows)):
        pq.write_table(table.slice(start, part_rows), split_dir / f"part-{part:05d}.parquet")
    by_id = {g["s1"]: g for g in groups}
    s1 = {"s1_id": s1_order, "truth_len": [], "n_cand": [], "retrieved_truth": []}
    for s1_id in s1_order:
        group = by_id.get(s1_id, {"rows": [], "truth_len": 0})
        s1["truth_len"].append(group["truth_len"] if labelled else -1)
        s1["n_cand"].append(len(group["rows"]))
        s1["retrieved_truth"].append(sum(r[1] for r in group["rows"]) if labelled else -1)
    pq.write_table(pa.table({"s1_id": pa.array(s1["s1_id"], pa.string()), "truth_len": pa.array(s1["truth_len"], pa.int32()),
                             "n_cand": pa.array(s1["n_cand"], pa.int16()),
                             "retrieved_truth": pa.array(s1["retrieved_truth"], pa.int32())}), split_dir / "s1.parquet")
    return np.asarray(logits, np.float32)


def make_groups(rng, s1_ids: list[str], counter: list[int]) -> tuple[list[dict], list[str]]:
    """Gold per S1 (unique targets, some not retrieved), negatives that are often another S1's gold."""
    gold = {}
    for s1 in s1_ids:
        gold[s1] = []
        for _ in range(int(rng.choice([0, 1, 1, 1, 2, 3]))):
            counter[0] += 1
            gold[s1].append(f"S{2 + counter[0] % 2}-{100000 + counter[0]}")
    all_gold = [t for ts in gold.values() for t in ts]
    groups, targets = [], set(all_gold)
    for s1 in s1_ids:
        if rng.random() < 0.05:
            continue
        rows = [(t, True) for t in gold[s1] if rng.random() < 0.85]
        seen = {t for t, _ in rows}
        for _ in range(int(rng.integers(0, 8))):
            if rng.random() < 0.4 and all_gold:
                t = all_gold[int(rng.integers(len(all_gold)))]
            else:
                counter[0] += 1
                t = f"S{2 + counter[0] % 2}-{900000 + counter[0]}"
            if t not in seen and t not in gold[s1]:
                seen.add(t)
                rows.append((t, False))
                targets.add(t)
        order = rng.permutation(len(rows))
        scored = [(rows[i][0], rows[i][1], float(3.0 * rows[i][1] - 1.5 + rng.normal(0, 1.2))) for i in order]
        groups.append({"s1": s1, "rows": scored, "truth_len": len(gold[s1])})
    rng.shuffle(groups)
    return groups, sorted(targets)


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def make_world(root: Path) -> dict:
    rng = np.random.default_rng(7)
    counter = [0]
    pairs, scores = root / "pairs", root / "ce" / "scores"
    scores.mkdir(parents=True)
    head = ["entity_id", "business_name", "business_address", "country"]
    train_rows, test_targets = [], set()
    for split, base, n in (("validation", 1000, 300), ("holdout", 5000, 300), ("test", 9000, 250)):
        s1_ids = [f"S1-{base + i}" for i in range(n)]
        groups, targets = make_groups(rng, s1_ids, counter)
        order = sorted(s1_ids)
        np.save(scores / f"{split}.npy", write_pairs(pairs / split, groups, order, split != "test"))
        if split == "test":
            test_targets = targets
            file_order = [s1_ids[i] for i in rng.permutation(n)]
            write_tsv(root / "test" / "test_source1.tsv", head,
                      [[s, f"Shop {s}", f"{i} Road", "France" if i % 3 == 0 else "US"] for i, s in enumerate(file_order)])
            test = {"groups": groups, "file_order": file_order}
        else:
            train_rows += [[s, f"Shop {s}", "", "India" if i % 2 else "US"] for i, s in enumerate(s1_ids)]
    for source in (2, 3):
        write_tsv(root / "test" / f"test_source{source}.tsv", head,
                  [[t, f"Biz {t}", "", "US"] for t in sorted(test_targets) if t.startswith(f"S{source}-")])
    write_tsv(root / "train" / "train_source1.tsv", head, train_rows)
    return {"pairs": pairs, "scores": scores, "test_dir": root / "test", "train_dir": root / "train"} | test


def read_lists(path: Path, header: str) -> list[tuple[str, list[str]]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == header and lines[-1] == ""
    out = []
    for line in lines[1:-1]:
        s1, ids = line.split("\t")
        out.append((s1, ids.split(",") if ids else []))
    return out


def test_loader_handles_part_boundaries_and_rejects_bad_layouts(tmp_path):
    groups = [{"s1": "S1-2", "rows": [("S2-1", True, 1.0), ("S3-2", False, 0.0), ("S2-3", False, -1.0)], "truth_len": 2},
              {"s1": "S1-1", "rows": [("S2-1", False, 0.5), ("S3-9", True, 2.0)], "truth_len": 1}]
    np.save(tmp_path / "validation.npy", write_pairs(tmp_path / "validation", groups, ["S1-1", "S1-2", "S1-3"], True, 2))
    d = load_split(tmp_path, tmp_path, "validation")
    assert d["starts"].tolist() == [0, 3, 5] and d["run_s1"].tolist() == [1, 0]
    assert d["g"].tolist() == [1, 1, 1, 0, 0] and d["n"] == 3
    assert d["t_code"][0] == d["t_code"][3] and d["checks"]["s1_without_rows"] == 1
    assert d["checks"]["truth_in_lists"] == 2 and d["checks"]["truth_total"] == 3
    split = [groups[0] | {"rows": groups[0]["rows"][:1]}, groups[1], groups[0] | {"rows": groups[0]["rows"][1:]}]
    np.save(tmp_path / "holdout.npy", write_pairs(tmp_path / "holdout", split, ["S1-1", "S1-2"], True, 2))
    with pytest.raises(ValueError, match="not contiguous"):
        load_split(tmp_path, tmp_path, "holdout")
    dup = [groups[0] | {"rows": groups[0]["rows"] + [("S2-1", False, 0.0)]}]
    np.save(tmp_path / "test.npy", write_pairs(tmp_path / "test", dup, ["S1-2"], True, 2))
    with pytest.raises(ValueError, match="duplicate"):
        load_split(tmp_path, tmp_path, "test")
    np.save(tmp_path / "validation.npy", np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="scores shape"):
        load_split(tmp_path, tmp_path, "validation")


def check_submission(world: dict, out: Path, one_owner_applied: bool) -> None:
    matching = read_lists(out / "output" / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids")
    candidate = read_lists(out / "output" / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids")
    assert [s for s, _ in matching] == world["file_order"] == [s for s, _ in candidate]
    expected = {g["s1"]: {t for t, _, _ in g["rows"]} for g in world["groups"]}
    claimed = []
    for (s1, matched), (_, cands) in zip(matching, candidate):
        assert set(cands) == expected.get(s1, set()) and len(cands) == len(set(cands))
        assert set(matched) <= set(cands) and len(matched) == len(set(matched))
        claimed += matched
    assert any(matched for _, matched in matching)
    if one_owner_applied:
        assert len(claimed) == len(set(claimed))


def test_end_to_end_tune_write_and_validator(tmp_path):
    world = make_world(tmp_path)
    out = tmp_path / "out"
    common = ["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--test-dir",
              str(world["test_dir"]), "--out", str(out), "--train-dir", str(world["train_dir"])]
    main(common + ["--stage", "all"])
    report = json.loads((out / "policy_report.json").read_text())
    names = [p["name"] for p in report["policies"]]
    assert {"P1", "P2", "P3", "P4(P1)", "P4(P2)", "P4(P3)"} <= set(names)
    assert report["chosen"]["name"] in names
    eligible = [p for p in report["policies"] if p["eligible"]]
    best = max(p["validation"]["macro_f05"] for p in eligible)
    assert report["chosen"]["validation"]["macro_f05"] >= best - 1e-5
    boot = report["holdout_bootstrap_chosen_minus_p1"]
    assert boot["reps"] == 1000 and boot["ci95"][0] <= boot["mean_diff"] <= boot["ci95"][1]
    assert report["oracle_macro_f05"]["validation"] >= best
    assert set(report["countries"]["holdout"]) == {"India", "US"}
    assert report["test"]["validator"]["exit_code"] == 0 and "PASS" in report["test"]["validator"]["stdout"]
    assert "--check-ids" in report["test"]["validator"]["command"] and report["test"]["validator"]["candidate_checked"]
    assert report["test"]["one_owner"]["targets_in_conflict"] > 0
    assert "France" in report["test"]["by_country"]
    assert {"x", "y"} <= set(json.loads((out / "calibrator.json").read_text()))
    assert "Validator: exit 0" in (out / "POLICY_REPORT.md").read_text()
    check_submission(world, out, report["chosen"]["one_owner"])

    # The write stage alone, forced onto the stacked one-owner variant.
    stacked = next(p for p in report["policies"] if p["name"] == "stacked-P4(P1)")
    report["chosen"] = {k: stacked[k] for k in ("name", "source", "family", "params", "one_owner", "complexity",
                                                "validation", "holdout")} | {"model": report["stacked_model"]}
    (out / "policy_report.json").write_text(json.dumps(report))
    main(common + ["--stage", "write"])
    report = json.loads((out / "policy_report.json").read_text())
    assert report["test"]["validator"]["passed"] and report["test"]["one_owner"]["applied"]
    check_submission(world, out, True)


def test_write_refuses_calibrator_from_another_tune_run_and_missing_validator(tmp_path):
    world = make_world(tmp_path)
    out = tmp_path / "out"
    common = ["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--test-dir",
              str(world["test_dir"]), "--out", str(out), "--train-dir", str(world["train_dir"])]
    with pytest.raises(SystemExit):
        main(common + ["--stage", "all", "--validator", str(tmp_path / "missing.py")])
    assert not out.exists()
    main(common + ["--stage", "tune"])
    report = json.loads((out / "policy_report.json").read_text())
    cal = json.loads((out / "calibrator.json").read_text())
    assert cal["run_id"] == report["calibrator"]["run_id"]
    (out / "calibrator.json").write_text(json.dumps(cal | {"run_id": "an-older-run"}))
    with pytest.raises(ValueError, match="different tune runs"):
        main(common + ["--stage", "write"])
    assert not (out / "output").exists()


def test_only_scored_top_k_rows_are_candidates_and_validation_uses_the_same_k(tmp_path):
    world = make_world(tmp_path)
    k = 3
    (world["scores"].parent / "test_k.json").write_text(json.dumps({"k": k}))
    test_logits = np.load(world["scores"] / "test.npy")
    ranks = np.concatenate([pq.read_table(p, columns=["filter_rank"]).column("filter_rank").to_numpy()
                            for p in sorted((world["pairs"] / "test").glob("part-*.parquet"))])
    test_logits[ranks >= k] = np.nan                                   # rows the cross-encoder did not score
    np.save(world["scores"] / "test.npy", test_logits)
    val = load_split(world["pairs"], world["scores"], "validation", max_rank=k)
    assert (val["filter_rank"] < k).all() and val["checks"]["rows_dropped_unscored"] > 0
    assert val["starts"][-1] == len(val["g"]) and len(val["run_s1"]) == len(val["starts"]) - 1
    out = tmp_path / "out"
    main(["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--test-dir",
          str(world["test_dir"]), "--out", str(out), "--train-dir", str(world["train_dir"]), "--stage", "all"])
    expected = {g["s1"]: {t for r, (t, _, _) in enumerate(g["rows"]) if r < k} for g in world["groups"]}
    matching = read_lists(out / "output" / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids")
    candidate = read_lists(out / "output" / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids")
    for (s1, matched), (_, cands) in zip(matching, candidate):
        assert set(cands) == expected.get(s1, set()) and set(matched) <= set(cands)
    assert json.loads((out / "policy_report.json").read_text())["test"]["validator"]["passed"]
