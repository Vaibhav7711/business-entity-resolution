"""Exact entity-level F-beta metrics for the challenge."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def parse_id_list(value: str | None) -> set[str]:
    """Parse a comma-separated challenge ID list.

    Empty strings and ``None`` represent an empty set. Whitespace surrounding an
    ID is ignored, but empty elements inside a non-empty list are rejected.
    """

    if value is None or value.strip() == "":
        return set()
    parts = [part.strip() for part in value.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"Malformed ID list: {value!r}")
    if len(parts) != len(set(parts)):
        raise ValueError(f"Duplicate ID in list: {value!r}")
    return set(parts)


def fbeta_set(
    truth: Iterable[str], prediction: Iterable[str], beta: float = 0.5
) -> float:
    """Score one Source 1 entity using the official singleton convention."""

    if beta <= 0:
        raise ValueError("beta must be positive")

    true_set = set(truth)
    pred_set = set(prediction)

    if not true_set:
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    beta_sq = beta * beta
    numerator = (1.0 + beta_sq) * tp
    denominator = numerator + fp + beta_sq * fn
    return numerator / denominator if denominator else 0.0


def macro_fbeta(
    truth_by_entity: Mapping[str, Iterable[str]],
    prediction_by_entity: Mapping[str, Iterable[str]],
    beta: float = 0.5,
) -> float:
    """Macro-average entity-level F-beta over exactly the ground-truth keys."""

    true_keys = set(truth_by_entity)
    pred_keys = set(prediction_by_entity)
    if true_keys != pred_keys:
        missing = sorted(true_keys - pred_keys)[:5]
        extra = sorted(pred_keys - true_keys)[:5]
        raise ValueError(
            "Prediction entity coverage differs from ground truth; "
            f"missing examples={missing}, extra examples={extra}"
        )
    if not true_keys:
        raise ValueError("Cannot score an empty evaluation set")

    return sum(
        fbeta_set(truth_by_entity[key], prediction_by_entity[key], beta=beta)
        for key in true_keys
    ) / len(true_keys)

