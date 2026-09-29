import numpy as np
import pyarrow.parquet as pq

from src.ce_cascade import main
from tests.test_ce_policy import make_world


def read(split_dir):
    parts = sorted(split_dir.glob("part-*.parquet"))
    rows = [pq.read_table(p).to_pydict() for p in parts]
    return {k: sum((r[k] for r in rows), []) for k in rows[0]}, pq.read_table(split_dir / "s1.parquet").to_pydict()


def test_cascade_keeps_top_n_by_round1_logit(tmp_path):
    world = make_world(tmp_path)
    test_logits = np.load(world["scores"] / "test.npy")
    test_logits[::7] = np.nan                                    # rows round 1 did not score are dropped
    np.save(world["scores"] / "test.npy", test_logits)
    out = tmp_path / "cascade"
    main(["--pairs-root", str(world["pairs"]), "--scores-dir", str(world["scores"]), "--out", str(out), "--n", "2"])
    for split in ("validation", "holdout", "test"):
        source, _ = read(world["pairs"] / split)
        logits = np.load(world["scores"] / f"{split}.npy")
        expected = {}
        for s1, t, label, logit in zip(source["s1_id"], source["t_id"], source["label"], logits):
            if np.isfinite(logit):
                expected.setdefault(s1, []).append((-float(logit), t, label))
        pairs, s1 = read(out / split)
        got = {}
        for s1_id, t, label, rank, score in zip(pairs["s1_id"], pairs["t_id"], pairs["label"], pairs["filter_rank"],
                                                pairs["filter_score"]):
            got.setdefault(s1_id, []).append((rank, t, label, score))
        for s1_id, rows in expected.items():
            top = sorted(rows)[:2]
            assert [(r, t, lab) for r, t, lab, _ in got[s1_id]] == [(i, t, lab) for i, (_, t, lab) in enumerate(top)]
        assert set(got) == set(expected)
        runs = [s for i, s in enumerate(pairs["s1_id"]) if i == 0 or pairs["s1_id"][i - 1] != s]
        assert len(runs) == len(set(runs))                       # S1 rows stay contiguous
        counts = {s: len(v) for s, v in got.items()}
        assert s1["n_cand"] == [counts.get(s, 0) for s in s1["s1_id"]]
