import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.dense_merge import main
from tests.test_ce_policy import make_world


def rows(split_dir, cols):
    t = pa.concat_tables([pq.read_table(p, columns=cols) for p in sorted(split_dir.glob("part-*.parquet"))])
    return t.to_pydict()


def test_dense_merge_new_and_augment(tmp_path):
    world = make_world(tmp_path)
    dense, rng = tmp_path / "dense", np.random.default_rng(0)
    gold_lines = ["source1_entity_id\tmatched_entity_ids"]
    for split in ("validation", "holdout", "test"):
        base = rows(world["pairs"] / split, ["s1_id", "t_id"])
        by_s1 = {}
        for s, t in zip(base["s1_id"], base["t_id"]):
            by_s1.setdefault(s, []).append(t)
        out = {"s1_id": [], "t_id": [], "dense_score": [], "dense_rank": []}
        for s, ts in by_s1.items():
            cands = ts[:2] + [f"S{2 + i % 2}-{7_000_000 + abs(hash((s, i))) % 1_000_000}" for i in range(4)]
            for r, t in enumerate(cands):
                out["s1_id"].append(s); out["t_id"].append(t); out["dense_score"].append(1 - r / 10); out["dense_rank"].append(r)
            if split != "test":
                gold_lines.append(f"{s}\t{cands[2]}")                 # the first new candidate is a true match
        (dense / split).mkdir(parents=True)
        pq.write_table(pa.table({k: pa.array(v) for k, v in out.items()}).cast(
            pa.schema([("s1_id", pa.string()), ("t_id", pa.string()), ("dense_score", pa.float32()), ("dense_rank", pa.int16())])),
            dense / split / "part-00000.parquet")
    (world["train_dir"] / "train_ground_truth.tsv").write_text("\n".join(gold_lines) + "\n")
    new = tmp_path / "new"
    main(["--stage", "new", "--base-root", str(world["pairs"]), "--dense-root", str(dense), "--new-root", str(new),
          "--train-dir", str(world["train_dir"]), "--max-rank", "5"])
    new_scores = tmp_path / "new_scores"
    new_scores.mkdir()
    for split in ("validation", "holdout", "test"):
        n = rows(new / split, ["s1_id", "t_id", "label", "filter_rank"])
        base = set(zip(*rows(world["pairs"] / split, ["s1_id", "t_id"]).values()))
        assert not set(zip(n["s1_id"], n["t_id"])) & base and max(n["filter_rank"]) < 5
        assert set(n["label"]) == ({-1} if split == "test" else {0, 1})
        np.save(new_scores / f"{split}.npy", np.arange(len(n["s1_id"]), dtype=np.float32) + 1000)
    aug = tmp_path / "aug"
    main(["--stage", "augment", "--base-root", str(world["pairs"]), "--dense-root", str(dense), "--new-root", str(new),
          "--base-scores", str(world["scores"]), "--new-scores", str(new_scores), "--aug-root", str(aug)])
    for split in ("validation", "holdout", "test"):
        a = rows(aug / split, ["s1_id", "t_id", "filter_rank"])
        scores = np.load(aug / "scores" / f"{split}.npy")
        base = rows(world["pairs"] / split, ["s1_id", "t_id"]); b_scores = np.load(world["scores"] / f"{split}.npy")
        n = rows(new / split, ["s1_id", "t_id"]); n_scores = np.load(new_scores / f"{split}.npy")
        expected = dict(zip(zip(base["s1_id"], base["t_id"]), b_scores)) | dict(zip(zip(n["s1_id"], n["t_id"]), n_scores))
        assert np.allclose(scores, [expected[p] for p in zip(a["s1_id"], a["t_id"])])
        runs = [s for i, s in enumerate(a["s1_id"]) if i == 0 or a["s1_id"][i - 1] != s]
        assert len(runs) == len(set(runs))
        for s in set(a["s1_id"]):                                            # base rows before new rows per S1
            ranks = [r for x, r in zip(a["s1_id"], a["filter_rank"]) if x == s]
            assert ranks == sorted(ranks, key=lambda r: r >= 40)
        with np.load(aug / "extra" / f"{split}.npz") as z:
            extra = z["X"]
        assert extra.shape == (len(a["s1_id"]), 3) and (extra[:, 2] == (np.asarray(a["filter_rank"]) >= 40)).all()
        s1 = pq.read_table(aug / split / "s1.parquet").to_pydict()
        counts = {}
        for s in a["s1_id"]:
            counts[s] = counts.get(s, 0) + 1
        assert s1["n_cand"] == [counts.get(s, 0) for s in s1["s1_id"]]


def test_augment_drops_base_rows_the_base_model_did_not_score(tmp_path):
    import json

    world = make_world(tmp_path)
    dense, new, new_scores, aug = tmp_path / "dense", tmp_path / "new", tmp_path / "new_scores", tmp_path / "aug"
    new_scores.mkdir()
    for split in ("validation", "holdout", "test"):
        base = rows(world["pairs"] / split, ["s1_id", "t_id"])
        s1 = sorted(set(base["s1_id"]))[:5]
        table = pa.table({"s1_id": pa.array(s1, pa.string()), "t_id": pa.array([f"S2-{8_000_000 + i}" for i in range(5)], pa.string()),
                          "dense_score": pa.array([0.5] * 5, pa.float32()), "dense_rank": pa.array([0] * 5, pa.int16())})
        (dense / split).mkdir(parents=True)
        pq.write_table(table, dense / split / "part-00000.parquet")
    (world["train_dir"] / "train_ground_truth.tsv").write_text("source1_entity_id\tmatched_entity_ids\n")
    main(["--stage", "new", "--base-root", str(world["pairs"]), "--dense-root", str(dense), "--new-root", str(new),
          "--train-dir", str(world["train_dir"]), "--max-rank", "5"])
    (world["scores"].parent / "test_k.json").write_text(json.dumps({"k": 2}))
    for split in ("validation", "holdout", "test"):
        n = rows(new / split, ["s1_id"])
        np.save(new_scores / f"{split}.npy", np.ones(len(n["s1_id"]), np.float32))
        b = np.load(world["scores"] / f"{split}.npy")
        ranks = np.asarray(rows(world["pairs"] / split, ["filter_rank"])["filter_rank"])
        b[ranks >= 2] = np.nan                                               # what a top-2 test run leaves unscored
        np.save(world["scores"] / f"{split}.npy", b)
    main(["--stage", "augment", "--base-root", str(world["pairs"]), "--dense-root", str(dense), "--new-root", str(new),
          "--base-scores", str(world["scores"]), "--new-scores", str(new_scores), "--aug-root", str(aug)])
    for split in ("validation", "holdout", "test"):
        a = rows(aug / split, ["filter_rank"])["filter_rank"]
        scores = np.load(aug / "scores" / f"{split}.npy")
        assert np.isfinite(scores).all() and len(scores) == len(a)
        assert all(r < 2 or r >= 40 for r in a) and sum(r >= 40 for r in a) == 5
