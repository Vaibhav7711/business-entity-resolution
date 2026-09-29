import json

import numpy as np
import pyarrow.parquet as pq

from src import ce_stack, stage2
from src.ce_context import CONTEXT_FEATURES
from tests.test_ce_policy import make_world


def test_aggregates_brute_force():
    g = np.array([0, 0, 0, 1, 1, 2])
    q = np.array([0.2, 0.9, 0.6, 0.3, 0.8, 0.5])
    X = stage2.aggregates(g, q)
    assert X[:, 0].tolist() == q.astype(np.float32).tolist()
    assert X[:, 1].tolist() == [2, 0, 1, 1, 0, 0]                          # rank within the list
    assert np.allclose(X[:, 2], [0.7, 0, 0.3, 0.5, 0, 0])                  # gap to the list best
    assert np.allclose(X[:, 3], [0.9, 0.6, 0.9, 0.8, 0.3, np.nan], equal_nan=True)
    assert np.allclose(X[:, 4], [1.7, 1.7, 1.7, 1.1, 1.1, 0.5]) and X[:, 5].tolist() == [2, 2, 2, 1, 1, 1]


def test_stage2_end_to_end_writes_a_valid_submission(tmp_path):
    world = make_world(tmp_path)
    ctx, extra = tmp_path / "ctx", tmp_path / "extra"
    ctx.mkdir(); extra.mkdir()
    rng = np.random.default_rng(3)
    for split in ("validation", "holdout", "test"):
        n = sum(pq.read_metadata(p).num_rows for p in sorted((world["pairs"] / split).glob("part-*.parquet")))
        np.savez(ctx / f"{split}.npz", X=rng.random((n, len(CONTEXT_FEATURES))).astype(np.float32), names=np.asarray(CONTEXT_FEATURES))
        np.savez(extra / f"{split}.npz", X=rng.random((n, 1)).astype(np.float32), names=np.asarray(["e"]))
    ce_stack.ROUNDS = 10
    out = tmp_path / "s2"
    stage2.main(["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--context-dir", str(ctx),
                 "--extra-dir", str(extra), "--test-dir", str(world["test_dir"]), "--out", str(out), "--threads", "1"])
    report = json.loads((out / "stack_report.json").read_text())
    assert report["test"]["validator"]["passed"] and 0 <= report["holdout_oof_macro_f05"] <= 1
