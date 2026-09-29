import json
import shutil

import numpy as np
import pyarrow.parquet as pq

from src.block_test import run_test as run_test_blocking
from src.k1_filter import Phase1CShards, Stores, build_stores, stage_report, stage_sample, stage_score, stage_train
from src.k2_features import build_features
from src.phase2a_env import ResourceGuard
from src.topk_export import export_fold0, export_test
from tests.test_k1_filter import k1_setup  # noqa: F401  (fixture)


def read_split(path):
    parts = sorted(path.glob("part-*.parquet"))
    rows = [pq.read_table(p).to_pydict() for p in parts]
    pairs = {k: sum((r[k] for r in rows), []) for k in rows[0]}
    return pairs, pq.read_table(path / "s1.parquet").to_pydict()


def test_topk_export_matches_k2_selection_and_contract(k1_setup):  # noqa: F811
    root, k1config, ctx, work1c, manifest, _ = k1_setup
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
    config["substitutions"].update(min_count=1, min_share=0.3)
    build_features(config, ctx, work1c, work, k1_out, keep_k=3, workers=1)       # K2's own top-3 lists

    out = root / "pairs"
    summary = export_fold0(config, work1c, k1_out, work, out, keep_k=3, workers=1, root=root)
    k2_keep = set()
    for path in sorted((work / "features").glob("chunk*.npz")):
        with np.load(path) as data:
            k2_keep |= {(ctx["ordered"][p], int(c)) for p, c in zip(data["s1_pos"], data["cand"])}
    from src.blocking import decode_id
    k2_keep = {(s, decode_id(c)) for s, c in k2_keep}
    gold = {}
    with open(root / "student_resource/dataset/train/train_ground_truth.tsv", encoding="utf-8") as file:
        next(file)
        for line in file:
            sid, ids = line.rstrip("\n").split("\t")
            gold[sid] = set(ids.split(",")) if ids else set()
    exported, n_s1 = set(), 0
    for split in ("train", "validation", "holdout"):
        pairs, s1 = read_split(out / split)
        low, high = config["split"][split]
        assert s1["s1_id"] == ctx["ordered"][low:high] and summary[split]["s1"] == high - low
        n_s1 += len(s1["s1_id"])
        last = {}
        for sid, tid, label, rank in zip(pairs["s1_id"], pairs["t_id"], pairs["label"], pairs["filter_rank"]):
            assert rank < 3 and last.get(sid, -1) < rank          # sorted by rank within S1
            last[sid] = rank
            exported.add((sid, tid))
            assert label == int(tid in gold.get(sid, set()))
        assert s1["truth_len"] == [len(gold.get(sid, set())) for sid in s1["s1_id"]]
        assert sum(pairs["label"]) <= sum(s1["truth_len"])
        counts = {sid: 0 for sid in s1["s1_id"]}
        for sid in pairs["s1_id"]:
            counts[sid] += 1
        assert [counts[sid] for sid in s1["s1_id"]] == s1["n_cand"]
    assert n_s1 == len(ctx["ordered"]) and exported == k2_keep

    # Test side: same selection code path through K3's driver, no labels.
    test_dir = root / "test_only"
    test_dir.mkdir()
    for i in (1, 2, 3):
        shutil.copy(root / f"student_resource/dataset/train/train_source{i}.tsv", test_dir / f"test_source{i}.tsv")
    p1c = json.loads((root / "configs/p1c.json").read_text())
    blocking = root / "test_blocking"
    n_test = sum(1 for _ in open(test_dir / "test_source1.tsv")) - 1
    last_shard = (n_test + 9) // 10 - 1
    run_test_blocking(p1c, root, test_dir, "0-7", blocking / "a" / "work", blocking / "a")
    run_test_blocking(p1c, root, test_dir, f"8-{last_shard}", blocking / "b" / "work", blocking / "b")
    k3 = json.loads(open("../../configs/k3_inference.json").read())
    k3.update(chunk_s1=5, workers=1, inputs={"phase1c_config": "configs/p1c.json", "train_dir": "unused"})
    tsum = export_test(k3, test_dir, blocking, k1_out, root / "k3_work", out, keep_k=3, workers=1, root=root)
    pairs, s1 = read_split(out / "test")
    assert tsum["test"]["s1"] == n_test == len(s1["s1_id"]) and set(pairs["label"]) == {-1}
    assert set(s1["truth_len"]) == {-1} and max(s1["n_cand"]) <= 3 and sum(s1["n_cand"]) == len(pairs["s1_id"])
