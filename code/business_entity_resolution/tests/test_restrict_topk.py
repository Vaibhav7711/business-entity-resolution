import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.restrict_topk import keep_mask, main
from tests.test_ce_policy import make_world


def rows(split_dir, cols):
    return pa.concat_tables([pq.read_table(p, columns=cols) for p in sorted(split_dir.glob("part-*.parquet"))]).to_pydict()


def test_keep_mask_takes_each_groups_top_k():
    codes = np.array([0, 0, 0, 1, 1, 2, 0])
    scores = np.array([0.1, 0.9, np.nan, 0.5, 0.7, 0.2, 0.9], np.float32)
    assert keep_mask(codes, scores, 2).tolist() == [False, True, False, True, True, True, True]


def test_cut_keeps_top_k_rows_and_aligned_arrays(tmp_path):
    world = make_world(tmp_path)
    feats = tmp_path / "feats"
    feats.mkdir()
    for split in ("validation", "holdout", "test"):
        n = len(np.load(world["scores"] / f"{split}.npy"))
        np.savez(feats / f"{split}.npz", X=np.arange(n, dtype=np.float32)[:, None], names=np.asarray(["row"]))
    out = tmp_path / "cut"
    main(["--pairs-root", str(world["pairs"]), "--rank-scores", str(world["scores"]), "--k", "2", "--out", str(out),
          "--scores-dirs", f"{world['scores']}:{out / 'scores'}", "--feature-dirs", f"{feats}:{out / 'feats'}"])
    for split in ("validation", "holdout", "test"):
        before = rows(world["pairs"] / split, ["s1_id", "t_id"])
        scores = np.load(world["scores"] / f"{split}.npy")
        after = rows(out / split, ["s1_id", "t_id"])
        kept_rows = np.load(out / "feats" / f"{split}.npz")["X"][:, 0].astype(int)
        assert [before["t_id"][i] for i in kept_rows] == after["t_id"]                    # aligned, order kept
        assert np.allclose(np.load(out / "scores" / f"{split}.npy"), scores[kept_rows])
        for s in set(before["s1_id"]):
            idx = [i for i, x in enumerate(before["s1_id"]) if x == s]
            top = sorted(idx, key=lambda i: (-scores[i], i))[:2]
            assert sorted(top) == sorted(i for i in kept_rows if before["s1_id"][i] == s)
        s1_before, s1_after = pq.read_table(world["pairs"] / split / "s1.parquet"), pq.read_table(out / split / "s1.parquet")
        assert s1_after.column("truth_len").to_pylist() == s1_before.column("truth_len").to_pylist()
        assert max(s1_after.column("n_cand").to_pylist()) <= 2
