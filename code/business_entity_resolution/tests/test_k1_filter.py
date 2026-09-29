import json

import numpy as np
import pytest

from src.evaluate_phase1c import run as run_phase1c
from src.k1_filter import (
    CHEAP_FEATURES, Phase1CShards, Stores, build_stores, load_context, oracle_f05, rank_within_s1, record_tokens,
    stage_report, stage_sample, stage_score, stage_train, token_hash, token_substitutions,
)
from src.phase2a_env import ResourceGuard
from tests.test_phase1c_evaluation import make_dataset, small_config


def test_tokens_strip_leading_zeros_and_keep_order():
    digits, names, words = record_tokens("eye care of maynard", "013102 hwy 115 maynard ar 0", {"digits": 4, "name": 8, "address": 10})
    assert digits == [token_hash("13102"), token_hash("115"), token_hash("0")]
    assert names == [token_hash(t) for t in ("eye", "care", "of", "maynard")]
    assert words == [token_hash(t) for t in ("hwy", "maynard", "ar")]


def test_rank_oracle_and_substitutions():
    rank = rank_within_s1(np.asarray([7, 7, 7, 9]), np.asarray([0.2, 0.9, 0.2, 0.1]), np.asarray([5, 6, 4, 1]))
    assert rank.tolist() == [2, 0, 1, 0]          # ties broken by candidate ID
    f = oracle_f05(np.asarray([0, 2, 1, 0]), np.asarray([0, 2, 2, 3]))
    assert f[0] == 1 and f[1] == 1 and f[3] == 0 and 0 < f[2] < 1
    assert token_substitutions("2217 8th street rockford il", "002217 eighth street rockford illinois") == [
        ("8th", "eighth"), ("8th", "illinois"), ("il", "eighth"), ("il", "illinois")]


@pytest.fixture()
def k1_setup(tmp_path):
    make_dataset(tmp_path, n_s1=160, seed=11)
    p1c = small_config()
    p1c["algorithm"]["shard_size"] = 10
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/p1c.json").write_text(json.dumps(p1c))
    result = run_phase1c(p1c, tmp_path, "full", require_gate=False)
    work1c = tmp_path / p1c["paths"]["work_dir"]
    manifest = {"state": json.loads((work1c / "state.json").read_text()),
                "shard_content_sha256": {str(path.relative_to(work1c)): json.loads(path.read_text())["content_sha256"]
                                         for path in sorted((work1c / "shards").rglob("*.json"))}}
    (tmp_path / "artifacts/phase1c").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts/phase1c/run_manifest.json").write_text(json.dumps(manifest))
    n = len(result["_ordered"])
    config = json.loads(open("../../configs/k1_filter.json").read())
    config["inputs"]["phase1c_config"] = "configs/p1c.json"
    config.update(chunk_s1=5, keep_top=10, k_grid=[1, 2, 3, 5, 10, 100], test_s1_count=1000,
                  split={"unit": "S1", "train": [0, 40], "validation": [40, 60], "holdout": [60, n]})
    config["sampling"].update(top_rrf_negatives=3, random_negatives=3)
    config["lightgbm"].update(min_data_in_leaf=2, num_boost_round=20, early_stopping_rounds=5, num_threads=1)
    config["gates"]["retention_target"] = 0.5
    ctx = load_context(config, tmp_path, work1c)
    ctx["train_dir"] = tmp_path / "student_resource/dataset/train"
    return tmp_path, config, ctx, work1c, manifest, result


def test_k1_pipeline_end_to_end(k1_setup):
    root, config, ctx, work1c, manifest, phase1c = k1_setup
    guard = ResourceGuard(0, 1e9, "resume")
    work, out = root / "k1_work", root / "k1_out"
    build_stores(config, root, work, ctx, guard)
    stores = Stores(work)
    shards = Phase1CShards(work1c, manifest)
    stage_sample(config, ctx, stores, shards, work, guard)
    stage_train(config, ctx, work, out, guard)
    assert {p.name for p in (out / "models").glob("filter_*.txt")} >= {"filter_fold0.txt", "filter_final.txt"}
    stage_score(config, ctx, stores, shards, work, out, guard)
    result = stage_report(config, ctx, work, out)
    for split in ("train", "validation", "holdout"):
        curve = result["by_split"][split]["all"]["curve"]
        retention = [curve[str(k)]["retention"] for k in config["k_grid"]]
        assert retention == sorted(retention) and retention[-1] == 1.0      # all gold kept once K >= candidates
        assert curve["100"]["oracle_macro_f05"] == pytest.approx(result["by_split"][split]["all"]["oracle_macro_f05_no_filter"])
    assert result["by_split"]["validation"]["all"]["s1"] == 20
    assert result["by_split"]["validation"]["all"]["retrieved_gold"] + result["by_split"]["holdout"]["all"]["retrieved_gold"] \
        + result["by_split"]["train"]["all"]["retrieved_gold"] == phase1c["metrics"]["selected_bce"]["retrieved_true_links"]
    assert result["decision"]["chosen_k"] is not None
    assert (out / "K1_REPORT.md").exists() and (out / "substitutions_train_gold.json").exists()
    with np.load(work / "topk/shard000.npz") as data:
        assert data["ids"].shape[1] == 10 and (data["counts"] <= 10).all()
    assert len(CHEAP_FEATURES) == len(set(CHEAP_FEATURES))


def test_phase1c_shard_tampering_is_detected(k1_setup):
    root, config, ctx, work1c, manifest, _ = k1_setup
    key = next(iter(manifest["shard_content_sha256"]))
    manifest["shard_content_sha256"][key] = "0" * 64
    with pytest.raises(RuntimeError, match="differs from the run manifest"):
        Phase1CShards(work1c, manifest).load(0, sorted({ctx["queries"]["country_key"][p] for p in range(10)}))
