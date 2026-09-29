import itertools

import numpy as np

from src.expected_f import best_set_sizes, decide_expected_f, poisson_binomial


def f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def brute_expected(p, k):
    n = len(p)
    total = 0.0
    for outcome in itertools.product((0, 1), repeat=n):
        prob = np.prod([pi if o else 1 - pi for pi, o in zip(p, outcome)])
        total += prob * f05(set(range(k)), {i for i, o in enumerate(outcome) if o})
    return total


def test_poisson_binomial_sums_to_one_and_matches_brute_force():
    p = np.array([[0.9, 0.2, 0.5], [0.0, 0.0, 0.0]])
    d = poisson_binomial(p)
    assert np.allclose(d.sum(1), 1) and np.isclose(d[1, 0], 1)
    assert np.isclose(d[0, 3], 0.9 * 0.2 * 0.5) and np.isclose(d[0, 0], 0.1 * 0.8 * 0.5)


def test_best_set_size_matches_brute_force_expected_f():
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(1, 6))
        p = np.sort(rng.random(n) ** rng.choice([0.3, 1, 3]))[::-1]
        k, e = best_set_sizes(p[None, :])
        brute = [brute_expected(p, kk) for kk in range(n + 1)]
        assert np.isclose(e[0], max(brute), atol=1e-9) and np.isclose(brute[k[0]], max(brute), atol=1e-9)


def test_decide_keeps_each_groups_top_k_star():
    g = np.array([0, 0, 0, 1, 1, 2])
    q = np.array([0.95, 0.9, 0.05, 0.4, 0.02, 0.97])
    pred = decide_expected_f(g, q, 3, width=4)
    assert pred.tolist() == [True, True, False, False, False, True]      # a lone 0.4 is not worth a guess
