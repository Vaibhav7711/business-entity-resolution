import json

import numpy as np

from src.k1_filter import Phase1CShards, Stores, build_stores, stage_report, stage_sample, stage_score, stage_train
from src.k2_experiments import load_split, run_experiments
from src.k2_features import ALL_FEATURES, build_canon_map, build_features, canonical, digit_key, fold_latin, script_code
from src.phase2a_env import ResourceGuard
from tests.test_k1_filter import k1_setup  # noqa: F401  (fixture)


def test_normalisation_helpers():
    assert fold_latin("café crème") == "cafe creme"
    assert fold_latin("लक्ष्मी बिल्डर्स") == "लक्ष्मी बिल्डर्स"      # Indic vowel signs untouched
    maps = build_canon_map({"address": [["8th", "eighth", 50], ["8th", "illinois", 10], ["il", "illinois", 40],
                                        ["rare", "other", 3]]}, min_count=30, min_share=0.5)
    assert canonical("2217 eighth street rockford il", maps["address"]) == canonical("2217 8th street rockford illinois", maps["address"])
    assert "rare" not in maps["address"]
    a = digit_key(np.asarray([[5, 9, 0, 0], [9, 5, 0, 0], [0, 0, 0, 0]], dtype=np.uint32))
    assert a[0] == a[1] and a[2] == 0
    assert script_code("acme") == 0 and script_code("স্মার্ট") != script_code("लक्ष्मी") != 0


def test_k2_end_to_end_on_synthetic_fold(k1_setup):  # noqa: F811
    root, k1config, ctx, work1c, manifest, phase1c = k1_setup
    guard = ResourceGuard(0, 1e9, "resume")
    work, k1_out = root / "work", root / "k1_out"
    build_stores(k1config, root, work, ctx, guard)
    stores, shards = Stores(work), Phase1CShards(work1c, manifest)
    stage_sample(k1config, ctx, stores, shards, work, guard)
    stage_train(k1config, ctx, work, k1_out, guard)
    stage_score(k1config, ctx, stores, shards, work, k1_out, guard)
    stage_report(k1config, ctx, work, k1_out)

    config = json.loads(open("../../configs/k2_matcher.json").read())
    config.update(chunk_s1=5, split=k1config["split"], workers=1, inputs=k1config["inputs"])
    config["filter"]["k_override"] = 3
    config["substitutions"].update(min_count=1, min_share=0.3)
    for key in ("lightgbm_ablation", "lightgbm_champion"):
        config[key].update(min_data_in_leaf=2, num_boost_round=20, early_stopping_rounds=5, num_threads=1)
    config["experiments"].update(E0_train_s1=20, ablation_train_s1=30, learning_curve=[10, 30, 40],
                                 oof_train_s1_per_fold=20, transfer_train_s1=40, bootstrap=100)
    config["thresholds"] = {"grid": 11, "empty_guard_grid": 3}
    build_features(config, ctx, work1c, work, k1_out, keep_k=3, workers=1)
    chunks = sorted((work / "features").glob("chunk*.npz"))
    assert len(chunks) == (len(ctx["ordered"]) + 4) // 5
    with np.load(chunks[0]) as data:
        assert data["X"].shape[1] == len(ALL_FEATURES)
        assert np.bincount(data["s1_pos"] - data["s1_positions"][0]).max() <= 3
        assert (data["s1_kept_truth"] <= data["s1_retrieved_truth"]).all()
    valid = load_split(work, config, ctx, "validation", sample=False)
    assert len(valid["ent_positions"]) == 20
    out = root / "k2_out"
    result = run_experiments(config, ctx, work, out, guard)
    for name in ("E0", "E1", "E2", "E3"):
        assert 0 <= result["ablations"][name]["validation"]["macro_f05"] <= 1
    champion = result["champion"]
    for split in ("validation", "holdout"):
        assert champion[split]["macro_f05"] <= champion[split]["oracle_macro_f05_kept"] + 1e-9
    assert (out / "models/champion.txt").exists() and (out / "models/champion_calibrator.json").exists()
    assert set(result["E4_reassignment"]["splits"]) == {"validation", "holdout"}
    assert "curve" in result["learning_curve"] and (out / "K2_REPORT.md").exists()
    budget = champion["error_budget_validation"]["categories"]
    assert "gold_dropped_by_filter" in budget and "fp_same_address_owned_by_other_s1" in budget


def test_k2_trains_on_extra_folds(k1_setup):  # noqa: F811
    import copy

    from src.evaluate_phase1c import run as run_phase1c
    from src.k1_filter import build_s1_store
    from src.k2_experiments import extra_fold_context

    root, k1config, ctx, work1c, manifest, _ = k1_setup
    guard = ResourceGuard(0, 1e9, "resume")
    work, k1_out = root / "work", root / "k1_out"
    build_stores(k1config, root, work, ctx, guard)
    stores, shards = Stores(work), Phase1CShards(work1c, manifest)
    stage_sample(k1config, ctx, stores, shards, work, guard)
    stage_train(k1config, ctx, work, k1_out, guard)
    stage_score(k1config, ctx, stores, shards, work, k1_out, guard)
    stage_report(k1config, ctx, work, k1_out)
    # Block fold 1 as an extra training fold and record its manifest, like the cloud script does.
    p1c = json.loads((root / "configs/p1c.json").read_text())
    p1c1 = copy.deepcopy(p1c)
    p1c1["algorithm"]["validation_fold"] = 1
    (root / "configs/phase1c_fold1.json").write_text(json.dumps(p1c1))
    work_k = root / "folds/fold1/work"
    run_phase1c(p1c1, root, "full", require_gate=False, work_dir=work_k)
    (root / "artifacts/phase1c_fold1").mkdir(parents=True, exist_ok=True)
    (root / "artifacts/phase1c_fold1/fold1_manifest.json").write_text(json.dumps({
        "state": json.loads((work_k / "state.json").read_text()),
        "shard_content_sha256": {str(p.relative_to(work_k)): json.loads(p.read_text())["content_sha256"]
                                 for p in sorted((work_k / "shards").rglob("*.json"))}}))
    config = json.loads(open("../../configs/k2_matcher.json").read())
    config.update(chunk_s1=5, split=k1config["split"], workers=1, inputs=k1config["inputs"], extra_train_folds=[1])
    config["filter"]["k_override"] = 3
    config["substitutions"].update(min_count=1, min_share=0.3)
    for key in ("lightgbm_ablation", "lightgbm_champion"):
        config[key].update(min_data_in_leaf=2, num_boost_round=20, early_stopping_rounds=5, num_threads=1)
    config["experiments"].update(E0_train_s1=20, ablation_train_s1=30, learning_curve=[10, 30, 40],
                                 oof_train_s1_per_fold=20, transfer_train_s1=40, bootstrap=100)
    config["thresholds"] = {"grid": 11, "empty_guard_grid": 3}
    build_features(config, ctx, work1c, work, k1_out, keep_k=3, workers=1)
    cfg1, ctx1 = extra_fold_context(config, 1, work_k, root)
    build_s1_store(cfg1, work / "stores_fold1", ctx1)
    build_features(cfg1, ctx1, work_k, work, k1_out, keep_k=3, workers=1, s1_dir=work / "stores_fold1",
                   features_dir="features_fold1", filter_all_final=True)
    extra = [{"fold": 1, "ctx": ctx1, "features_dir": "features_fold1", "offset": 10_000_000}]
    train = load_split(work, config, ctx, "train", sample=True, extra=extra)
    fold0_only = load_split(work, config, ctx, "train", sample=True)
    assert len(np.unique(train["s1_pos"])) == len(np.unique(fold0_only["s1_pos"])) + len(ctx1["ordered"])
    assert (np.diff(train["ent_positions"]) > 0).all()
    result = run_experiments(config, ctx, work, root / "k2_extra_out", guard, extra)
    assert str(40 + len(ctx1["ordered"])) in result["learning_curve"]["curve"]
    assert result["champion"]["training"]["rows"] == len(train["label"])
