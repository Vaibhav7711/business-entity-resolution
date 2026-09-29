"""Official entity-level macro F0.5 over candidate-pair predictions, vectorized.

Every evaluated Source 1 entity counts once. Its truth size includes gold links
the blocker never retrieved, so blocking misses lower recall exactly as in the
official metric. Singletons score 1 for an empty prediction and 0 otherwise.
"""

from __future__ import annotations

import numpy as np

BETA_SQ = 0.25


def entity_fbeta(group: np.ndarray, label: np.ndarray, predicted: np.ndarray,
                 truth_len: np.ndarray) -> np.ndarray:
    """Per-entity F0.5 for pair rows (group = entity index 0..n-1).

    ``truth_len`` is the full gold-set size per entity, retrieved or not.
    """
    n = len(truth_len)
    tp = np.bincount(group, weights=(predicted & label), minlength=n)
    n_pred = np.bincount(group, weights=predicted, minlength=n)
    fp = n_pred - tp
    fn = truth_len - tp
    denominator = (1 + BETA_SQ) * tp + fp + BETA_SQ * fn
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(denominator > 0, (1 + BETA_SQ) * tp / denominator, 0.0)
    singleton = truth_len == 0
    score[singleton] = (n_pred[singleton] == 0).astype(float)
    return score


def group_max(group: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    result = np.full(n, -np.inf)
    np.maximum.at(result, group, values)
    return result


def apply_policy(group: np.ndarray, score: np.ndarray, n: int, threshold: float,
                 empty_threshold: float | None = None, best: np.ndarray | None = None) -> np.ndarray:
    """Pairs at or above ``threshold``; an entity whose best score is below
    ``empty_threshold`` predicts the empty set (singleton-aware guard)."""
    predicted = score >= threshold
    if empty_threshold is not None and empty_threshold > threshold:
        best = group_max(group, score, n) if best is None else best
        predicted &= (best >= empty_threshold)[group]
    return predicted


def sweep(group: np.ndarray, label: np.ndarray, score: np.ndarray, truth_len: np.ndarray,
          thresholds: np.ndarray, empty_thresholds: np.ndarray | None = None) -> dict:
    """Grid search of (threshold, empty_threshold) by macro F0.5. Ties keep the
    higher (more conservative) threshold pair."""
    n = len(truth_len)
    best_scores = group_max(group, score, n)
    grid = []
    best = None
    for threshold in thresholds:
        empties = [None] + [e for e in (empty_thresholds if empty_thresholds is not None else []) if e > threshold]
        for empty in empties:
            predicted = apply_policy(group, score, n, threshold, empty, best_scores)
            macro = float(entity_fbeta(group, label, predicted, truth_len).mean())
            row = {"threshold": float(threshold), "empty_threshold": None if empty is None else float(empty),
                   "macro_f05": macro}
            grid.append(row)
            if best is None or macro > best["macro_f05"] + 1e-12 or (
                    abs(macro - best["macro_f05"]) <= 1e-12 and threshold >= best["threshold"]):
                best = row
    return {"best": best, "grid": grid}


def plateau(grid: list[dict], best: dict, tolerance: float = 0.001) -> dict:
    """Threshold range whose macro F0.5 is within ``tolerance`` of the best (no guard)."""
    near = [row["threshold"] for row in grid if row["empty_threshold"] is None
            and row["macro_f05"] >= best["macro_f05"] - tolerance]
    return {"tolerance": tolerance, "min_threshold": min(near) if near else None,
            "max_threshold": max(near) if near else None}
