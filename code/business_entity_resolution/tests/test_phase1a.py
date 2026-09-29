from src.blocking import decode_id, encode_id, exact_name_lookup, union_candidates
from src.evaluate_blocking import ranked_union, recall_at, summarize_route
from src.normalization import normalize_text


def test_normalization_preserves_scripts_and_separates_punctuation():
    assert normalize_text("  ＡＣＭＥ—Co.\tLLC  ") == "acme co llc"
    assert normalize_text("राम  मार्केटिंग, दिल्ली") == "राम मार्केटिंग दिल्ली"
    assert normalize_text(None) == ""


def test_exact_lookup_retains_distinct_ids_with_same_text():
    ids = [encode_id("S2-11"), encode_id("S2-12"), encode_id("S3-13")]
    assert exact_name_lookup(["acme"], ["acme"] * 3, ids) == [ids]
    assert [decode_id(value) for value in ids] == ["S2-11", "S2-12", "S3-13"]


def test_union_deduplicates_by_id_and_ranked_union_retains_all():
    assert union_candidates([1, 2], [2, 3], [3, 4]) == [1, 2, 3, 4]
    assert set(ranked_union([1, 2], [2, 3], [3, 4])) == {1, 2, 3, 4}
    assert len(ranked_union([1, 2], [2, 3], [3, 4])) == 4


def test_recall_counts_distinct_true_ids_and_singleton_every_match():
    assert recall_at([1, 1, 2], [1, 2, 3]) == 2
    assert recall_at([1, 1, 2], [1, 2, 3], 2) == 1
    metrics = summarize_route(
        [[1, 2], []], [[1, 2, 3], []], ["X", "Y"], [{1}, set()], [{3}, set()]
    )
    assert metrics["positive_link_candidate_recall"] == 2 / 3
    assert metrics["entities_with_every_true_match_pct"] == 50
    assert metrics["positive_entities_with_every_true_match_pct"] == 0
    assert metrics["recall_at"]["10"] == 2 / 3
    assert metrics["zero_candidate_rate"] == 0.5
