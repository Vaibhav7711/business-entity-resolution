import numpy as np
from rapidfuzz import fuzz

from src.sib_context import group_features


def brute(names, logits, top=3):
    order = [int(i) for i in np.argsort(-np.asarray(logits), kind="stable")]
    out = []
    for i in range(len(names)):
        sib = [j for j in order if j != i][:top]
        sims = [fuzz.token_set_ratio(names[i], names[j]) for j in sib]
        out.append([max(sims), float(np.mean(sims))] if sims else [np.nan, np.nan])
    return np.asarray(out, np.float32)


def test_group_features_match_brute_force():
    rng = np.random.default_rng(0)
    words = ["atlantic", "micro", "electronics", "inc", "roman", "lyrium", "llc", "raoyal", "dental", "kirkland"]
    for n in (1, 2, 3, 4, 7, 12):
        names = [" ".join(rng.choice(words, rng.integers(1, 4))) for _ in range(n)]
        logits = rng.normal(size=n).astype(np.float32)
        logits[rng.random(n) < 0.2] = np.nan                               # unscored rows rank last
        filled = np.where(np.isfinite(logits), logits, -np.inf)
        assert np.allclose(group_features(names, logits), brute(names, filled), equal_nan=True, atol=1e-4), n
