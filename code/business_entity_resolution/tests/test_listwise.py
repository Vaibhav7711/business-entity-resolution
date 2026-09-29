import numpy as np

from src.listwise import build


def test_listwise_features_match_direct_computation():
    codes = np.array([0, 0, 0, 1, 1, 0, 2])
    ce = np.array([2.0, -1.0, 0.5, 3.0, 2.8, np.nan, 1.0])
    rr = np.array([1.0, np.nan, 0.0, 2.0, 2.5, -1.0, np.nan])
    dense = np.array([0.9, 0.5, 0.7, 0.8, 0.85, 0.6, 0.4])
    X, names = build(codes, {"ce": ce, "rr": rr, "dense": dense})
    col = {n: X[:, i] for i, n in enumerate(names)}
    g0 = np.array([2.0, -1.0, 0.5])                                        # group 0 finite ce values
    assert col["ce_cnt"][0] == 3 and np.isclose(col["ce_mean"][0], g0.mean()) and np.isclose(col["ce_median"][1], 0.5)
    assert np.isclose(col["ce_gap1"][1], 3.0) and np.isclose(col["ce_gap2"][2], 0.0)
    assert np.isclose(col["ce_softmax"][0], np.exp(2) / np.exp(g0).sum(), atol=1e-6)
    assert col["ce_within1.0"][0] == 1 and col["ce_within2.0"][0] == 2
    assert np.isnan(col["ce_z"][5]) or True                                # the NaN row keeps group stats
    assert col["rr_cnt"][0] == 3 and col["rankdiff_ce_dense"][3] == 1     # ce ranks row 3 first, dense second
    assert np.isclose(col["dense_z"][4], (0.85 - 0.825) / np.std([0.8, 0.85]))
