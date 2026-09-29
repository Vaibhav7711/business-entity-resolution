from array import array

import numpy as np

from src.evaluate_phase1b import evaluate_config, pareto_masks


def test_incremental_candidate_metrics_and_reduction():
    baseline = {
        "exact_indptr": np.asarray([0, 1, 1]),
        "exact_values": np.asarray([2], dtype=np.uint64),
        "name_ids": np.zeros((2, 2, 1), dtype=np.uint64),
        "address_ids": np.zeros((2, 2, 1), dtype=np.uint64),
    }
    extras = {
        "name200": np.zeros((2, 2, 1), dtype=np.uint64),
        "word": np.zeros((2, 2, 1), dtype=np.uint64),
        "rare": [array("Q", [4]), array("Q")],
        "digit": [array("Q"), array("Q")],
        "suffix": [array("Q"), array("Q")],
    }
    kwargs = dict(
        baseline=baseline, extras=extras, truths=[{2, 4}, set()],
        countries=["X", "Y"], non_ascii=[{4}, set()], missing=[{4}, set()],
        pool_sizes={"X": 10, "Y": 20},
    )
    base = evaluate_config(0, **kwargs)
    plus = evaluate_config(4, **kwargs)
    assert base["positive_link_recall"] == 0.5
    assert plus["positive_link_recall"] == 1
    assert base["positive_s1_with_every_true_match_pct"] == 0
    assert plus["positive_s1_with_every_true_match_pct"] == 100
    assert plus["non_ascii_recall"] == 1
    assert plus["missing_target_address_recall"] == 1
    assert plus["candidate_count"]["total"] - base["candidate_count"]["total"] == 1
    assert plus["candidate_reduction_ratio"] == 1 - 2 / 30
    assert pareto_masks({0: base, 4: plus}) == [0, 4]
