import copy
import json

import numpy as np

from src.evaluate_phase1c import run as run_phase1c
from src.k1_filter import Phase1CShards, Stores, build_s1_store, build_stores, stage_report, stage_sample, stage_score, stage_train
from src.k2_experiments import extra_fold_context, load_split
from src.k2_export import export_fold, load_export
from src.k2_features import build_features
from src.phase2a_env import ResourceGuard
from tests.test_k1_filter import k1_setup  # noqa: F401  (fixture)


def test_r4_export_matches_k2_training_and_evaluation_rows(k1_setup):  # noqa: F811
    root, k1config, ctx, work1c, manifest, _ = k1_setup
    guard = ResourceGuard(0, 1e9, "resume")
    work, k1_out = root / "work", root / "k1_out"
    build_stores(k1config, root, work, ctx, guard)
    stores, shards = Stores(work), Phase1CShards(work1c, manifest)
    stage_sample(k1config, ctx, stores, shards, work, guard)
    stage_train(k1config, ctx, work, k1_out, guard)
    stage_score(k1config, ctx, stores, shards, work, k1_out, guard)
    stage_report(k1config, ctx, work, k1_out)
    # Extra fold 1, blocked where R4's runner links it: artifacts/phase1c_fold1/{work, fold1_manifest.json}.
    p1c1 = copy.deepcopy(json.loads((root / "configs/p1c.json").read_text()))
    p1c1["algorithm"]["validation_fold"] = 1
    (root / "configs/phase1c_fold1.json").write_text(json.dumps(p1c1))
    work_k = root / "artifacts/phase1c_fold1/work"
    run_phase1c(p1c1, root, "full", require_gate=False, work_dir=work_k)
    (root / "artifacts/phase1c_fold1/fold1_manifest.json").write_text(json.dumps({
        "state": json.loads((work_k / "state.json").read_text()),
        "shard_content_sha256": {str(p.relative_to(work_k)): json.loads(p.read_text())["content_sha256"]
                                 for p in sorted((work_k / "shards").rglob("*.json"))}}))
    config = json.loads(open("../../configs/k2_matcher.json").read())
    config.update(chunk_s1=5, split=k1config["split"], workers=1, inputs=k1config["inputs"], extra_train_folds=[1])
    config["substitutions"].update(min_count=1, min_share=0.3)
    config["train_sampling"].update(top_filter_negatives=1, random_negatives=1)

    # K2's own path: full feature chunks, sampled at load time.
    build_features(config, ctx, work1c, work, k1_out, keep_k=4, workers=1)
    cfg1, ctx1 = extra_fold_context(config, 1, work_k, root)
    build_s1_store(cfg1, work / "stores_fold1", ctx1)
    build_features(cfg1, ctx1, work_k, work, k1_out, keep_k=4, workers=1, s1_dir=work / "stores_fold1",
                   features_dir="features_fold1", filter_all_final=True)
    extra = [{"fold": 1, "ctx": ctx1, "features_dir": "features_fold1", "offset": 10_000_000}]
    k2_train = load_split(work, config, ctx, "train", sample=True, extra=extra)

    # R4's path: sampled at feature time, written compactly, read back.
    out = root / "r4"
    info0 = export_fold(config, ctx, 0, work1c, work, k1_out, 4, out / "fold0", root)
    info1 = export_fold(config, ctx, 1, work1c, work, k1_out, 4, out / "fold1", root)
    assert info0["s1"] == len(ctx["ordered"]) and info1["s1"] == len(ctx1["ordered"])
    fold0, fold1 = load_export(out / "fold0", "train"), load_export(out / "fold1")
    names = ("X", "label", "s1_pos", "cand", "weight", "country", "ent_positions", "ent_truth_len", "ent_kept_truth",
             "ent_retrieved_truth", "ent_cand_count", "ent_country")
    for name in names:
        combined = np.concatenate([fold0[name], fold1[name]])
        assert np.array_equal(combined, k2_train[name], equal_nan=combined.dtype.kind == "f"), name
    assert (fold1["weight"] > 1).any()                                  # natural-rate weights survive the export
    assert np.array_equal(fold1["ent_key"], ctx1["s1_num"])
    for split in ("validation", "holdout"):
        mine, theirs = load_export(out / "fold0", split), load_split(work, config, ctx, split, sample=False)
        for name in names:
            assert np.array_equal(mine[name], theirs[name], equal_nan=mine[name].dtype.kind == "f"), (split, name)
