import numpy as np

from src.evaluate_phase1c import atomic_savez
from src.phase2a_pairs import FEATURES, TargetStore, compute_features, s1_pairs, sample_train_rows


def fake_loaded():
    """One S1 (local row 0) with S2/S3 route shards in Phase 1C format."""
    dense = lambda ids, scores: {"ids": np.asarray([ids], np.uint32), "scores": np.asarray([scores], np.float32),
                                 "counts": np.asarray([len(ids)], np.uint16), "positions": np.asarray([7])}
    ragged = lambda ids, scores=None: {"indptr": np.asarray([0, len(ids)]), "ids": np.asarray(ids, np.uint32),
                                       "positions": np.asarray([7]),
                                       **({"scores": np.asarray(scores, np.float32)} if scores is not None else {})}
    loaded = {}
    for source, base in ((2, 10), (3, 11)):
        loaded["exact_name", source] = ragged([base] if source == 2 else [])
        loaded["name_char", source] = dense([base, base + 2], [0.9, 0.5])
        loaded["address_char", source] = dense([base + 4], [0.7])
        loaded["name_word", source] = dense([base], [0.8])
        loaded["rare_name", source] = ragged([base + 6], [0.4])
        loaded["suffix_exact", source] = ragged([])
    return loaded


def test_s1_pairs_unique_labelled_with_route_evidence():
    row = s1_pairs(7, 0, fake_loaded(), truth=np.asarray([10, 15, 99]), rrf_constant=60)
    assert sorted(row["cand"].tolist()) == row["cand"].tolist()
    assert row["cand"].tolist() == [10, 11, 12, 13, 14, 15, 16, 17]
    assert row["label"].tolist() == [c in (10, 15) for c in row["cand"]]
    assert row["retrieved_truth"] == 2          # gold 99 was never retrieved: a blocking miss
    i10 = 0
    assert row["route_bits"][i10] == 0b001011   # exact + name_char + name_word
    assert row["rank_name_char"][i10] == 1 and np.isnan(row["score_rare_name"][i10])
    assert row["rrf_rank"][i10] == 0


def test_train_sampling_is_deterministic_and_keeps_all_positives():
    s1_pos = np.repeat([0, 1], 40)
    label = np.zeros(80, bool); label[[3, 50, 51]] = True
    pairs = {"s1_pos": s1_pos, "label": label, "rrf_rank": np.tile(np.arange(40), 2)}
    a = sample_train_rows(pairs, seed=1, chunk=0, hard=5, random_count=5)
    b = sample_train_rows(pairs, seed=1, chunk=0, hard=5, random_count=5)
    assert np.array_equal(a, b)
    assert {3, 50, 51} <= set(a.tolist())
    assert len(a) == (1 + 10) + (2 + 10)


def test_features_mark_missing_address_as_not_applicable(tmp_path):
    names = ["acme traders", "acme traders ltd"]
    addresses = ["12 mg road", ""]
    (tmp_path / "targets_name.bin").write_bytes("".join(names).encode())
    (tmp_path / "targets_address.bin").write_bytes("".join(addresses).encode())
    atomic_savez(tmp_path / "targets_meta.npz", {
        "sorted_ids": np.asarray([4, 5], np.uint32), "order": np.asarray([0, 1]),
        "name_offsets": np.cumsum([0] + [len(n.encode()) for n in names]),
        "address_offsets": np.cumsum([0] + [len(a.encode()) for a in addresses]),
        "non_ascii": np.asarray([False, False]), "address_missing": np.asarray([False, True]),
        "name_missing": np.asarray([False, False])})
    store = TargetStore(tmp_path)
    pairs = {"cand": np.asarray([4, 5], np.uint32), "route_bits": np.asarray([1, 2], np.uint8),
             "s1_pos": np.asarray([0, 0], np.int32), "rrf": np.asarray([0.03, 0.02], np.float32),
             "rrf_rank": np.asarray([0, 1]), "cand_count": np.asarray([2, 2]),
             "s1_positions": np.asarray([0]), "s1_exact_hits": np.asarray([1]),
             **{f"score_{r}": np.asarray([np.nan, 0.5], np.float32) for r in ("name_char", "address_char", "name_word", "rare_name")},
             **{f"rank_{r}": np.asarray([0, 1], np.int16) for r in ("name_char", "address_char", "name_word", "rare_name")}}
    s1 = {"name": ["acme traders"], "address": ["12 m g road"], "address_missing": np.asarray([False]),
          "non_ascii": np.asarray([False])}
    X = compute_features(pairs, np.asarray([0, 1]), s1, store)
    col = {name: X[:, i] for i, name in enumerate(FEATURES)}
    assert X.shape == (2, len(FEATURES))
    assert col["name_exact"].tolist() == [1, 0] and col["name_suffix_exact"].tolist() == [1, 1]
    assert col["digits_exact"][0] == 1
    assert np.isnan(col["addr_ratio"][1]) and col["tgt_addr_missing"][1] == 1   # missing != disagreement
    assert np.isnan(col["rank_name_char"][0]) and col["rank_name_char"][1] == 1
    assert col["is_s3"].tolist() == [0, 1]
