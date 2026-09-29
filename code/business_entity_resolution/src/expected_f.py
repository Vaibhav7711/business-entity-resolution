"""Per-S1 set decisions that maximise expected F0.5 (Jansche 2007; Nan Ye et al. 2012), replacing a global threshold.

For one S1 with candidate probabilities p (sorted descending, independent Bernoulli, as calibrated by the stacker),
predicting the top-k set has expected F0.5 = sum_{a,b} P(A = a) P(B = b) f(a, b, k), where A ~ Poisson-binomial of the
top-k probabilities (true positives), B ~ Poisson-binomial of the rest (missed links), and
f = 1.25 a / (1.25 a + 0.25 b + (k - a)) with the empty-set convention of the metric (k = 0 scores 1 only when there
is no true link at all; a non-empty prediction for an S1 without links scores 0). ``k*`` = argmax over k = 0..n.
All S1 are processed at once on padded (S1, n) arrays; ``extra_missed`` adds an expected count of true links outside
the list to the recall denominator (gold the candidate generation missed).
"""

from __future__ import annotations

import numpy as np


def poisson_binomial(p: np.ndarray) -> np.ndarray:
    """(N, m) probabilities -> (N, m + 1) distribution of the number of successes (padding rows use p = 0)."""
    n, m = p.shape
    dist = np.zeros((n, m + 1))
    dist[:, 0] = 1.0
    for j in range(m):
        pj = p[:, [j]]
        dist[:, 1:j + 2] = dist[:, 1:j + 2] * (1 - pj) + dist[:, 0:j + 1] * pj
        dist[:, 0] *= (1 - p[:, j])
    return dist


def f05_table(k: int, n: int) -> np.ndarray:
    """f(a, b) for predicting k items: a true positives among them, b true links not predicted."""
    a = np.arange(n + 1)[:, None].astype(float)
    b = np.arange(n + 1)[None, :].astype(float)
    if k == 0:
        return ((a == 0) & (b == 0)).astype(float)
    denom = 1.25 * a + 0.25 * b + (k - a)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(a > 0, 1.25 * a / denom, 0.0)
    f[a[:, 0] > k, :] = 0.0                                               # impossible (a <= k)
    return f


def best_set_sizes(p: np.ndarray, extra_missed: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """p: (N, n) candidate probabilities sorted descending per row (pad with 0). Returns (k*, expected F at k*)."""
    n_rows, n = p.shape
    best_k = np.zeros(n_rows, np.int64)
    best_e = np.full(n_rows, -1.0)
    for k in range(n + 1):
        a_dist = poisson_binomial(p[:, :k]) if k else np.ones((n_rows, 1))
        b_dist = poisson_binomial(p[:, k:]) if k < n else np.ones((n_rows, 1))
        a_full = np.zeros((n_rows, n + 1)); a_full[:, :a_dist.shape[1]] = a_dist
        b_full = np.zeros((n_rows, n + 1)); b_full[:, :b_dist.shape[1]] = b_dist
        table = f05_table(k, n)
        if extra_missed and k:                                             # expected gold outside the list
            a = np.arange(n + 1)[:, None].astype(float); b = np.arange(n + 1)[None, :].astype(float) + extra_missed
            with np.errstate(divide="ignore", invalid="ignore"):
                table = np.where(a > 0, 1.25 * a / (1.25 * a + 0.25 * b + (k - a)), 0.0)
            table[a[:, 0] > k, :] = 0.0
        e = np.einsum("na,nb,ab->n", a_full, b_full, table)
        better = e > best_e
        best_k[better], best_e[better] = k, e[better]
    return best_k, best_e


def decide_expected_f(g: np.ndarray, q: np.ndarray, n_groups: int, width: int = 10, extra_missed: float = 0.0) -> np.ndarray:
    """Boolean decision per row: each S1 predicts its top-k* candidates by q (rows grouped by ``g``)."""
    order = np.lexsort((-q, g))
    gs = g[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]) if len(order) else np.zeros(0, np.int64)
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    keep = rank < width
    padded = np.zeros((n_groups, width))
    padded[gs[keep], rank[keep]] = q[order][keep]
    k_star, _ = best_set_sizes(padded, extra_missed)
    pred_sorted = rank < k_star[gs]
    pred = np.zeros(len(q), bool)
    pred[order] = pred_sorted
    return pred
