import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.rev_context import main


def dense_dir(path, rows):
    path.mkdir(parents=True)
    s, t, v, r = zip(*rows)
    pq.write_table(pa.table({"s1_id": pa.array(s, pa.string()), "t_id": pa.array(t, pa.string()),
                             "dense_score": pa.array(v, pa.float32()), "dense_rank": pa.array(r, pa.int16())}),
                   path / "part-00000.parquet")


def brute(rows, corpus, own):
    out = []
    for (s, t), o in zip(rows, own):
        others = [v for (s2, t2, v, _) in corpus if t2 == t and s2 != s]
        best = max(others) if others else np.nan
        margin = o - best if np.isfinite(o) and others else np.nan
        top = np.nan if not np.isfinite(o) else float(not others or o >= best)
        out.append([best, margin, len(others), top])
    return np.asarray(out, np.float32)


def test_competition_features_match_brute_force(tmp_path):
    rng = np.random.default_rng(0)
    folds = ["source1_entity_id\tfold"] + [f"S1-{i}\t{0 if i <= 10 else 1 + i % 4}" for i in range(1, 41)]
    (tmp_path / "folds.tsv").write_text("\n".join(folds) + "\n")
    targets = [f"S{2 + k % 2}-{100 + k}" for k in range(12)]
    train_corpus = [(f"S1-{i}", t, float(rng.random()), r) for i in range(1, 11) for r, t in
                    enumerate(rng.choice(targets, 5, replace=False))]
    test_s1 = [f"S1-{i}" for i in range(500, 540)]
    test_corpus = [(s, t, float(rng.random()), r) for s in test_s1 for r, t in enumerate(rng.choice(targets, 4, replace=False))]
    test_corpus.append(("S1-500", "S2-100", 0.9, 60))                       # rank beyond top_k: ignored
    dense_dir(tmp_path / "train_dense_a", train_corpus[:25])
    dense_dir(tmp_path / "train_dense_b", train_corpus[25:])
    dense_dir(tmp_path / "test_dense", test_corpus)
    for split, s1s in (("validation", ["S1-1", "S1-2", "S1-3"]), ("test", test_s1[:6])):
        rows = [(s, t) for s in s1s for t in targets[:7]]
        (tmp_path / "aug" / split).mkdir(parents=True)
        pq.write_table(pa.table({"s1_id": pa.array([s for s, _ in rows], pa.string()),
                                 "t_id": pa.array([t for _, t in rows], pa.string())}), tmp_path / "aug" / split / "part-00000.parquet")
        own = np.where(rng.random(len(rows)) < 0.8, rng.random(len(rows)), np.nan).astype(np.float32)
        (tmp_path / "extra").mkdir(exist_ok=True)
        np.savez(tmp_path / "extra" / f"{split}.npz", X=np.column_stack([own, np.zeros(len(rows))]).astype(np.float32),
                 names=np.asarray(["dense_score", "dense_rank"]))
    main(["--pairs-root", str(tmp_path / "aug"), "--extra-dir", str(tmp_path / "extra"), "--train-corpus",
          str(tmp_path / "train_dense_a"), str(tmp_path / "train_dense_b"), "--test-corpus", str(tmp_path / "test_dense"),
          "--folds", str(tmp_path / "folds.tsv"), "--out", str(tmp_path / "rev"), "--top-k", "50", "--splits", "validation", "test"])
    keep_rng = np.random.default_rng(20260927)
    all_test = sorted(set(test_s1))
    kept = set(np.asarray(all_test, dtype=object)[keep_rng.random(len(all_test)) < 10 / 40].tolist())
    sampled = [row for row in test_corpus if row[0] in kept and row[3] < 50]
    for split, corpus in (("validation", train_corpus), ("test", sampled)):
        rows = pq.read_table(tmp_path / "aug" / split / "part-00000.parquet").to_pydict()
        pairs = list(zip(rows["s1_id"], rows["t_id"]))
        own = np.load(tmp_path / "extra" / f"{split}.npz")["X"][:, 0]
        got = np.load(tmp_path / "rev" / f"{split}.npz")["X"]
        assert np.allclose(got, brute(pairs, corpus, own), equal_nan=True), split
