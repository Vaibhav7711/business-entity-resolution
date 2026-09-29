import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src import ce_stack
from src.ce_context import CONTEXT_FEATURES, main as context_main
from tests.test_ce_policy import make_world, read_lists, write_tsv

HEAD = ["entity_id", "business_name", "business_address", "country"]


def test_context_counts_on_a_small_corpus(tmp_path):
    train = tmp_path / "train"
    write_tsv(train / "train_source1.tsv", HEAD, [["S1-1", "Acme Ltd", "5 Main Rd", "US"], ["S1-2", "Acme Ltd", "5 Main Rd", "US"],
                                                  ["S1-3", "Bolt", "9 High St", "US"], ["S1-4", "Acme Ltd", "", "India"]])
    write_tsv(train / "train_source2.tsv", HEAD, [["S2-10", "ACME LTD", "", "US"], ["S2-11", "Other", "5 Main Rd", "US"]])
    write_tsv(train / "train_source3.tsv", HEAD, [["S3-20", "Acme Ltd", "5 main rd", "US"], ["S3-21", "Bolt", "", "US"]])
    write_tsv(tmp_path / "test" / "test_source1.tsv", HEAD, [["S1-9", "X", "", "US"]] * 1)
    folds = tmp_path / "folds.tsv"
    folds.write_text("source1_entity_id\tfold\nS1-1\t0\nS1-2\t1\nS1-3\t2\nS1-4\t0\n")
    split = tmp_path / "pairs" / "validation"
    split.mkdir(parents=True)
    rows = [("S1-1", "S2-10"), ("S1-1", "S2-11"), ("S1-1", "S3-20"), ("S1-3", "S3-21")]
    for part, chunk in enumerate((rows[:2], rows[2:])):                     # a list spanning two parts
        pq.write_table(pa.table({"s1_id": [r[0] for r in chunk], "t_id": [r[1] for r in chunk]}), split / f"part-{part:05d}.parquet")
    context_main(["--pairs-root", str(tmp_path / "pairs"), "--out", str(tmp_path / "ctx"), "--train-dir", str(train),
                  "--test-dir", str(tmp_path / "test"), "--folds", str(folds), "--splits", "validation"])
    with np.load(tmp_path / "ctx" / "validation.npz") as z:
        X = dict(zip(CONTEXT_FEATURES, z["X"].T))
    # counting subset = fold-0 S1 (S1-1, S1-4) + 0 sampled others (target size 1 < 2 fold-0 S1)
    assert X["s1_same_name"].tolist() == [1, 1, 1, 0]            # S1-2 (fold 1) not counted; India Acme is another country
    assert X["tg_same_name"].tolist() == [2, 1, 2, 1] and X["s1_same_tname"].tolist() == [1, 0, 1, 0]
    assert X["t_addr_empty"].tolist() == [1, 0, 0, 1] and X["name_eq"].tolist() == [1, 0, 1, 1]
    assert X["addr_eq"].tolist() == [0, 1, 1, 0] and X["tg_same_addr"].tolist() == [0, 2, 2, 0]
    assert X["list_same_tname"].tolist() == [2, 1, 2, 1]         # S2-10 and S3-20 share a name across the part boundary


def test_stack_end_to_end_with_validator(tmp_path):
    world = make_world(tmp_path)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    rng = np.random.default_rng(0)
    for split in ("validation", "holdout", "test"):
        n = sum(pq.read_metadata(p).num_rows for p in sorted((world["pairs"] / split).glob("part-*.parquet")))
        np.savez(ctx / f"{split}.npz", X=rng.random((n, len(CONTEXT_FEATURES))).astype(np.float32),
                 names=np.asarray(CONTEXT_FEATURES))
    out = tmp_path / "stack"
    ce_stack.ROUNDS = 20
    ce_stack.main(["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--context-dir", str(ctx),
                   "--test-dir", str(world["test_dir"]), "--out", str(out), "--threads", "1"])
    report = json.loads((out / "stack_report.json").read_text())
    assert report["test"]["validator"]["passed"], report["test"]["validator"]["stdout"]
    matching = read_lists(out / "output" / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids")
    candidate = read_lists(out / "output" / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids")
    assert [s for s, _ in matching] == world["file_order"]
    for (s1, m), (_, c) in zip(matching, candidate):
        assert set(m) <= set(c)
    assert 0 <= report["holdout"]["macro_f05"] <= 1 and (out / "stacker.txt").exists()


def test_stack_takes_several_extra_feature_dirs_and_checks_their_rows(tmp_path):
    import pytest

    world = make_world(tmp_path)
    ctx, extra_a, extra_b = tmp_path / "ctx", tmp_path / "extra_a", tmp_path / "extra_b"
    rng = np.random.default_rng(1)
    for directory in (ctx, extra_a, extra_b):
        directory.mkdir()
    for split in ("validation", "holdout", "test"):
        n = sum(pq.read_metadata(p).num_rows for p in sorted((world["pairs"] / split).glob("part-*.parquet")))
        np.savez(ctx / f"{split}.npz", X=rng.random((n, len(CONTEXT_FEATURES))).astype(np.float32),
                 names=np.asarray(CONTEXT_FEATURES))
        np.savez(extra_a / f"{split}.npz", X=rng.random((n, 2)).astype(np.float32), names=np.asarray(["a1", "a2"]))
        np.savez(extra_b / f"{split}.npz", X=rng.random((n, 1)).astype(np.float32), names=np.asarray(["b1"]))
    ce_stack.ROUNDS = 10
    out = tmp_path / "stack"
    ce_stack.main(["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--context-dir", str(ctx),
                   "--extra-dir", str(extra_a), str(extra_b), "--out", str(out), "--no-test", "--threads", "1"])
    features = json.loads((out / "stack_report.json").read_text())["features"]
    assert features[-3:] == ["a1", "a2", "b1"]
    np.savez(extra_b / "validation.npz", X=np.zeros((3, 1), np.float32), names=np.asarray(["b1"]))
    with pytest.raises(ValueError, match="extra features"):
        ce_stack.main(["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--context-dir",
                       str(ctx), "--extra-dir", str(extra_a), str(extra_b), "--out", str(out), "--no-test"])


def test_pooled_stack_writes_valid_test_output(tmp_path):
    world = make_world(tmp_path)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    rng = np.random.default_rng(2)
    for split in ("validation", "holdout", "test"):
        n = sum(pq.read_metadata(p).num_rows for p in sorted((world["pairs"] / split).glob("part-*.parquet")))
        np.savez(ctx / f"{split}.npz", X=rng.random((n, len(CONTEXT_FEATURES))).astype(np.float32),
                 names=np.asarray(CONTEXT_FEATURES))
    ce_stack.ROUNDS = 10
    out = tmp_path / "pooled"
    ce_stack.main(["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--context-dir", str(ctx),
                   "--test-dir", str(world["test_dir"]), "--out", str(out), "--threads", "1", "--pooled", "--bag", "2"])
    report = json.loads((out / "stack_report.json").read_text())
    assert report["mode"].startswith("pooled") and report["test"]["validator"]["passed"] and report["bag"] == 2
    assert (out / "stacker.txt").exists() and (out / "stacker.txt.bag1").exists()
    assert 0 <= report["validation_oof_macro_f05"] <= 1 and 0 <= report["holdout_oof_macro_f05"] <= 1
