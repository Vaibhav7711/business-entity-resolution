import random

import numpy as np

from src.metrics import fbeta_set, macro_fbeta
from src.phase2a_eval import apply_policy, entity_fbeta, sweep


def to_sets(group, ids, predicted, n):
    sets = [set() for _ in range(n)]
    for g, i, p in zip(group, ids, predicted):
        if p:
            sets[g].add(i)
    return sets


def test_entity_fbeta_matches_official_scorer_including_blocking_misses():
    rng = random.Random(3)
    n = 300
    group, ids, label, truth_sets = [], [], [], []
    for g in range(n):
        truth = {f"T{g}-{k}" for k in range(rng.choice([0, 0, 1, 2, 4]))}
        retrieved = {t for t in truth if rng.random() < 0.8}          # some gold never retrieved
        candidates = sorted(retrieved | {f"N{g}-{k}" for k in range(rng.randint(0, 6))})
        truth_sets.append(truth)
        for c in candidates:
            group.append(g); ids.append(c); label.append(c in truth)
    group = np.asarray(group); label = np.asarray(label)
    truth_len = np.asarray([len(t) for t in truth_sets])
    predicted = np.asarray([rng.random() < 0.5 for _ in ids])
    ours = entity_fbeta(group, label, predicted, truth_len)
    pred_sets = to_sets(group, ids, predicted, n)
    official = [fbeta_set(truth_sets[g], pred_sets[g]) for g in range(n)]
    assert np.allclose(ours, official)
    keys = [f"S1-{g}" for g in range(n)]
    assert np.isclose(ours.mean(), macro_fbeta(dict(zip(keys, truth_sets)), dict(zip(keys, pred_sets))))


def test_singleton_empty_is_one_and_any_match_is_zero():
    group = np.asarray([0, 0, 1])
    label = np.asarray([False, False, True])
    truth_len = np.asarray([0, 1])
    assert entity_fbeta(group, label, np.asarray([False, False, True]), truth_len).tolist() == [1.0, 1.0]
    assert entity_fbeta(group, label, np.asarray([True, False, True]), truth_len).tolist() == [0.0, 1.0]
    # Entity with gold but zero candidates (total blocking miss) scores 0 whatever we predict.
    assert entity_fbeta(np.asarray([0]), np.asarray([False]), np.asarray([False]), np.asarray([0, 2]))[1] == 0.0


def test_empty_guard_and_sweep_prefer_empty_for_weak_singletons():
    group = np.asarray([0, 0, 1])
    label = np.asarray([True, False, False])
    score = np.asarray([0.9, 0.3, 0.4])
    truth_len = np.asarray([1, 0])
    guarded = apply_policy(group, score, 2, threshold=0.35, empty_threshold=0.5)
    assert guarded.tolist() == [True, False, False]
    result = sweep(group, label, score, truth_len, np.asarray([0.2, 0.35, 0.5]), np.asarray([0.5, 0.8]))
    assert result["best"]["macro_f05"] == 1.0
