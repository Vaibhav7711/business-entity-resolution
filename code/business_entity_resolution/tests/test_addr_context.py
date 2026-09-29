import numpy as np
from rapidfuzz import fuzz

from src.addr_context import pair_features, street_text


def test_street_text_drops_numbers_and_normalises():
    assert "17" not in street_text("17 Rue de Bruges, Bordeaux")
    assert street_text("") == ""


def test_pair_features_match_rapidfuzz_and_mark_empty_addresses():
    a = [street_text(x) for x in ("17 Rue de Bruges, Bordeaux", "17 Rue de Bruges, Bordeaux", "", "44 Gorski Street, Amsterdam, NY")]
    b = [street_text(x) for x in ("17 R. DE BRUGES, BORDEAUX", "17 Rue Xaintrailles, Bordeaux", "12 Main St", "44 GORSKI ST, AMSTERDAM, NY")]
    got = pair_features(a, b)
    for i in (0, 1, 3):
        assert np.isclose(got[i, 0], fuzz.token_set_ratio(a[i], b[i]), atol=1e-3)
        assert np.isclose(got[i, 1], fuzz.token_sort_ratio(a[i], b[i]), atol=1e-3)
        assert np.isclose(got[i, 2], fuzz.partial_ratio(a[i], b[i]), atol=1e-3)
    assert np.isnan(got[2]).all()                                        # an empty address -> NaN
    assert got[0, 1] > got[1, 1]                                         # same street beats another street
