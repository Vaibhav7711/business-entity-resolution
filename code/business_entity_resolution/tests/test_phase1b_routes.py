from src.phase1b_routes import (
    address_digits,
    name_tokens,
    rare_token_lookup,
    strip_legal_suffix,
    stripped_exact_lookup,
    word_tfidf_topk_rows,
)


def test_legal_suffix_uses_only_statement_list_and_keeps_root():
    assert strip_legal_suffix("acme private limited") == "acme"
    assert strip_legal_suffix("acme pvt ltd") == "acme"
    assert strip_legal_suffix("acme corporation") == "acme"
    assert strip_legal_suffix("limited") == "limited"
    assert strip_legal_suffix("acme llc") == "acme llc"


def test_rare_name_token_lookup_is_frequency_bounded_and_deduped():
    rows = rare_token_lookup(
        ["acme uncommon", "common"],
        ["uncommon shop", "uncommon common", "common shop", "common store"],
        [2, 4, 6, 8], tokenize=name_tokens,
        max_document_frequency=2, max_query_tokens=2, k=10,
    )
    assert rows[0] == [2, 4]
    assert rows[1] == []


def test_address_digits_and_stripped_exact_lookup():
    assert address_digits("12-34, 560001, A5") == {"12", "34", "560001"}
    assert stripped_exact_lookup(
        ["acme ltd", "blue corp"],
        ["acme limited", "acme", "blue corporation", "blue roof"],
        [2, 4, 6, 8],
    ) == [[2, 4], [6]]


def test_word_tfidf_sparse_topk():
    rows = list(word_tfidf_topk_rows(
        ["acme rare name"], ["acme rare name", "different words"],
        [2, 4], k=2, n_features=1024,
        max_document_frequency=1, query_batch_size=1, threads=1,
    ))
    assert rows[0][0][0] == 2
