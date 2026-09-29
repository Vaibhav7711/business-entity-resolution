import numpy as np

from src.country_rule import list_rank, row_thresholds


def test_list_rank_orders_by_q_within_group():
    g = np.array([0, 0, 1, 0, 1])
    q = np.array([0.2, 0.9, 0.4, 0.5, 0.8])
    assert list_rank(g, q).tolist() == [2, 0, 1, 1, 0]


def test_row_thresholds_only_touch_country_rows():
    g = np.array([0, 0, 1, 1, 2, 2])
    q = np.array([0.6, 0.3, 0.9, 0.7, 0.6, 0.55])
    in_country = np.array([True, True, True, True, False, False])
    base = q >= 0.8                                                      # group 0 and 2 empty, group 1 has a match
    thr = row_thresholds(g, q, in_country, 0.8, base, t_country=0.7, t_empty_top=0.5)
    assert thr.tolist() == [0.5, 0.7, 0.7, 0.7, 0.8, 0.8]                # empty-list top of the country at 0.5


def test_row_thresholds_default_keeps_pooled_rule():
    g = np.array([0, 0, 1])
    q = np.array([0.6, 0.3, 0.9])
    thr = row_thresholds(g, q, np.ones(3, bool), 0.8, q >= 0.8)
    assert thr.tolist() == [0.8, 0.8, 0.8]
