import json

import numpy as np

from src.phase2a_env import ResourceGuard
from src.phase2a_pairs import Layout, build_features, build_pairs, build_texts, data_qa
from src.phase2a_report import make_readme
from src.phase2a_train import build_context, run_model
from src.evaluate_phase1c import run as run_phase1c
from tests.test_phase1c_evaluation import make_dataset, small_config


def test_phase2a_pipeline_end_to_end_on_synthetic_benchmark(tmp_path):
    make_dataset(tmp_path, n_s1=160, seed=11)
    p1c = small_config()
    p1c["algorithm"]["shard_size"] = 10
    p1c["operational"]["benchmark_shards"] = 3
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/p1c.json").write_text(json.dumps(p1c))
    assert run_phase1c(p1c, tmp_path, "benchmark")["status"] == "complete"
    config = json.loads(open("../../configs/phase2a_benchmark.json").read())
    config["inputs"]["phase1c_config"] = "configs/p1c.json"
    config.update(chunk_s1=5, split={"unit": "S1", "train": [0, 20], "validation": [20, 25], "holdout": [25, 30]})
    config["sampling"].update(hard_negatives_per_s1=3, random_negatives_per_s1=3)
    config["models"]["lightgbm"].update(min_data_in_leaf=2, num_boost_round=20, early_stopping_rounds=5)
    config["thresholds"] = {"grid": 11, "empty_guard_grid": 3}
    layout = Layout(tmp_path / "artifacts/phase2a")
    guard = ResourceGuard(0, 1e9, "resume")
    build_pairs(config, tmp_path, layout, guard)
    build_texts(config, tmp_path, layout, guard)
    build_features(config, tmp_path, layout, guard)
    qa = data_qa(config, layout)
    assert qa["splits"]["validation"]["s1"] == 5 and qa["splits"]["train"]["feature_rows"] > 0
    context = build_context(config, layout, tmp_path)
    runs = {model: run_model(model, config, layout, guard, context) for model in ("rules", "logistic", "lightgbm")}
    assert runs["logistic"]["reproducible_refit_identical"]
    for result in runs.values():
        for split in ("validation", "holdout"):
            assert 0 <= result[split]["macro_f05"] <= result["oracle_ceiling"][split]["macro_f05"] + 1e-9
    assert runs["rules"]["empty_prediction_baseline"]["holdout"]["macro_f05"] >= 0
    readme = make_readme(config, qa, runs, "configs/p2a.json", "sha-test")
    assert "Holdout F0.5" in readme and "lightgbm" in readme
