import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.ce_join import list_features, main
from tests.test_ce_policy import make_world


def rows(split_dir, cols):
    t = pa.concat_tables([pq.read_table(p, columns=cols) for p in sorted(split_dir.glob("part-*.parquet"))])
    return t.to_pydict()


def test_join_aligns_by_pair_and_marks_unscored_rows(tmp_path):
    world = make_world(tmp_path)
    dst = tmp_path / "dst"
    for split in ("validation", "holdout", "test"):
        base = pa.concat_tables([pq.read_table(p) for p in sorted((world["pairs"] / split).glob("part-*.parquet"))])
        sub = base.take(pa.array(np.flatnonzero(np.arange(base.num_rows) % 3 != 1)))   # a subset, S1 grouping kept
        extra = sub.slice(0, 1)                                                          # a pair the source never had
        extra = extra.set_column(extra.schema.get_field_index("t_id"), "t_id",
                                 pa.array(["S3-99999991"], extra.schema.field("t_id").type))
        sub = pa.concat_tables([extra, sub])
        (dst / split).mkdir(parents=True)
        pq.write_table(sub, dst / split / "part-00000.parquet")
        pq.write_table(pq.read_table(world["pairs"] / split / "s1.parquet"), dst / split / "s1.parquet")
    out = tmp_path / "out"
    main(["--src-root", str(world["pairs"]), "--src-scores", str(world["scores"]), "--dst-root", str(dst),
          "--out", str(out), "--name", "r1b"])
    for split in ("validation", "holdout", "test"):
        src = rows(world["pairs"] / split, ["s1_id", "t_id"])
        logits = np.load(world["scores"] / f"{split}.npy")
        lookup = dict(zip(zip(src["s1_id"], src["t_id"]), logits))
        d = rows(dst / split, ["s1_id", "t_id"])
        with np.load(out / f"{split}.npz") as z:
            X, names = z["X"], list(z["names"])
        assert names == ["r1b_logit", "r1b_rank", "r1b_gap"] and len(X) == len(d["s1_id"])
        expected = np.asarray([lookup.get(p, np.nan) for p in zip(d["s1_id"], d["t_id"])], np.float32)
        assert np.isnan(X[0, 0]) and np.isnan(expected[0])
        assert np.allclose(X[:, 0], expected, equal_nan=True)


def test_list_features_rank_and_gap():
    logits = np.asarray([1.0, 3.0, np.nan, 2.0, -1.0, np.nan], np.float32)
    group = np.asarray([0, 0, 0, 0, 1, 1])
    rank, gap = list_features(logits, group)
    assert rank.tolist() == [2, 0, 3, 1, 0, 1]
    assert np.allclose(gap, [2.0, 0.0, np.nan, 1.0, 0.0, np.nan], equal_nan=True)


def test_join_masks_source_rows_beyond_the_test_k_on_every_split(tmp_path):
    import json

    world = make_world(tmp_path)
    (world["scores"].parent / "test_k.json").write_text(json.dumps({"k": 2}))
    out = tmp_path / "out"
    main(["--src-root", str(world["pairs"]), "--src-scores", str(world["scores"]), "--dst-root", str(world["pairs"]),
          "--out", str(out)])
    for split in ("validation", "holdout", "test"):
        ranks = np.asarray(rows(world["pairs"] / split, ["filter_rank"])["filter_rank"])
        with np.load(out / f"{split}.npz") as z:
            assert (np.isnan(z["X"][:, 0]) == (ranks >= 2)).all()
