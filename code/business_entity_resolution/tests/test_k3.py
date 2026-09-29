import csv
import json
import shutil

from src.block_test import run_test as run_test_blocking
from src.k1_filter import Phase1CShards, Stores, build_stores, stage_report, stage_sample, stage_score, stage_train
from src.k2_experiments import run_experiments
from src.k2_features import build_features
from src.k3_inference import load_models, load_test_context, run_validator, score_all, write_outputs
from src.k2_features import density_table
from src.phase2a_env import ResourceGuard
from tests.test_k1_filter import k1_setup  # noqa: F401  (fixture)


def read_lists(path):
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file, delimiter="\t"))
    return rows[0], {row[0]: (row[1].split(",") if len(row) > 1 and row[1] else []) for row in rows[1:]}


def test_k3_end_to_end_produces_validator_passing_submission(k1_setup):  # noqa: F811
    root, k1config, ctx, work1c, manifest, _ = k1_setup
    guard = ResourceGuard(0, 1e9, "resume")
    # K1 + K2 on the synthetic training fold.
    work, k1_out, k2_out = root / "work", root / "k1_out", root / "k2_out"
    build_stores(k1config, root, work, ctx, guard)
    stores, shards = Stores(work), Phase1CShards(work1c, manifest)
    stage_sample(k1config, ctx, stores, shards, work, guard)
    stage_train(k1config, ctx, work, k1_out, guard)
    stage_score(k1config, ctx, stores, shards, work, k1_out, guard)
    stage_report(k1config, ctx, work, k1_out)
    k2 = json.loads(open("../../configs/k2_matcher.json").read())
    k2.update(chunk_s1=5, split=k1config["split"], workers=1, inputs=k1config["inputs"])
    k2["substitutions"].update(min_count=1, min_share=0.3)
    for key in ("lightgbm_ablation", "lightgbm_champion"):
        k2[key].update(min_data_in_leaf=2, num_boost_round=20, early_stopping_rounds=5, num_threads=1)
    k2["experiments"].update(E0_train_s1=20, ablation_train_s1=30, learning_curve=[10, 30, 40],
                             oof_train_s1_per_fold=20, transfer_train_s1=40, bootstrap=100)
    k2["thresholds"] = {"grid": 11, "empty_guard_grid": 3}
    build_features(k2, ctx, work1c, work, k1_out, keep_k=4, workers=1)
    run_experiments(k2, ctx, work, k2_out, guard)
    (k2_out / "filter_choice.json").write_text(json.dumps({"keep_k": 4}))
    shutil.copy(work / "canon_map.json", k2_out / "models" / "canon_map.json")

    # Synthetic "test" split: copies of the sources with no ground truth, blocked in two shard ranges.
    test_dir = root / "test_only"
    test_dir.mkdir()
    for i in (1, 2, 3):
        shutil.copy(root / f"student_resource/dataset/train/train_source{i}.tsv", test_dir / f"test_source{i}.tsv")
    p1c = json.loads((root / "configs/p1c.json").read_text())
    blocking = root / "test_blocking"
    n_test = sum(1 for _ in open(test_dir / "test_source1.tsv")) - 1
    last = (n_test + 9) // 10 - 1
    run_test_blocking(p1c, root, test_dir, "0-7", blocking / "a" / "work", blocking / "a")
    run_test_blocking(p1c, root, test_dir, f"8-{last}", blocking / "b" / "work", blocking / "b")

    k3 = json.loads(open("../../configs/k3_inference.json").read())
    k3.update(chunk_s1=5, workers=1, inputs={"phase1c_config": "configs/p1c.json", "train_dir": "unused"})
    tctx = load_test_context(k3, root, test_dir)
    work3, out3 = root / "k3_work", root / "k3_out"
    build_stores(k3, root, work3, tctx, guard, targets_dir=test_dir, prefix="test")
    density_table(Stores(work3), work3)
    models = load_models(k1_out, k2_out)
    score_all(k3, tctx, work3, blocking, models, workers=1, shards=list(range(last + 1)))
    summary = write_outputs(k3, tctx, work3, out3, models, test_dir)
    validator = run_validator(out3, test_dir)
    assert validator["passed"], validator["output_tail"]
    c_header, candidates = read_lists(out3 / "output/candidate_pairs.tsv")
    m_header, matches = read_lists(out3 / "output/matching_results.tsv")
    assert c_header == ["source1_entity_id", "candidate_entity_ids"] and m_header == ["source1_entity_id", "matched_entity_ids"]
    assert len(candidates) == len(matches) == n_test
    assert all(set(matches[s1]) <= set(candidates[s1]) for s1 in matches)
    assert max(len(v) for v in candidates.values()) <= 4
    assert summary["labels_read"] is False and summary["candidate_pairs"] == sum(map(len, candidates.values()))
