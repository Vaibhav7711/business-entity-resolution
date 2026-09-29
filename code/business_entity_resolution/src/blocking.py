"""Bounded sparse candidate retrieval helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sparse_dot_topn import sp_matmul_topn


def encode_id(entity_id: str) -> int:
    source, number = entity_id.split("-", 1)
    if source not in {"S2", "S3"}:
        raise ValueError(f"Unexpected target source: {source}")
    return (int(number) << 1) | (source == "S3")


def decode_id(value: int) -> str:
    return f"S{3 if value & 1 else 2}-{value >> 1}"


def union_candidates(*routes: list[int]) -> list[int]:
    """Preserve first occurrence and retain distinct target IDs."""
    return list(dict.fromkeys(value for route in routes for value in route))


def exact_name_lookup(query_names: list[str], target_names: list[str], target_ids: list[int]) -> list[list[int]]:
    wanted = set(query_names) - {""}
    lookup: dict[str, list[int]] = defaultdict(list)
    for name, target_id in zip(target_names, target_ids):
        if name in wanted:
            lookup[name].append(target_id)
    return [lookup.get(name, []).copy() if name else [] for name in query_names]


class HashedTfidfIndex(NamedTuple):
    vectorizer: HashingVectorizer
    tfidf: TfidfTransformer
    keep: np.ndarray
    corpus_t: object


def fit_hashed_tfidf(
    target_texts: list[str], *, analyzer: str, ngram_range: tuple[int, int],
    n_features: int, max_document_frequency: float, token_pattern: str | None = None,
) -> HashedTfidfIndex:
    """Fit one target corpus once so query batches can be scored separately.

    The hash mapping is fixed, while IDF is fitted independently for each
    source/country/field corpus. Very common features are removed from both sides.
    """
    options = dict(analyzer=analyzer, ngram_range=ngram_range, n_features=n_features,
                   alternate_sign=False, norm=None, dtype=np.float32)
    if token_pattern is not None:
        options["token_pattern"] = token_pattern
    vectorizer = HashingVectorizer(**options)
    corpus = vectorizer.transform(target_texts).tocsr()
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    tfidf.fit(corpus)
    # Hashed document frequency follows from smoothed IDF: log((1+n)/(1+df))+1.
    n_docs = corpus.shape[0]
    df = (n_docs + 1) / np.exp(tfidf.idf_ - 1) - 1
    keep = df <= max_document_frequency * n_docs
    corpus.data *= keep[corpus.indices]
    corpus.eliminate_zeros()
    corpus = tfidf.transform(corpus, copy=False)
    corpus_t = corpus.T.tocsr()
    del corpus
    return HashedTfidfIndex(vectorizer, tfidf, keep, corpus_t)


def query_hashed_tfidf(index: HashedTfidfIndex, query_texts: list[str], *, k: int, threads: int):
    """Bounded top-k sparse product; each query row is scored independently."""
    query = index.vectorizer.transform(query_texts).tocsr()
    query.data *= index.keep[query.indices]
    query.eliminate_zeros()
    query = index.tfidf.transform(query, copy=False)
    return sp_matmul_topn(query, index.corpus_t, top_n=k, threshold=0.0,
                          sort=True, n_threads=threads)


def sparse_topk_rows(
    query_texts: list[str],
    target_texts: list[str],
    target_ids: list[int],
    *,
    k: int,
    n_features: int,
    ngram_range: tuple[int, int],
    max_document_frequency: float,
    query_batch_size: int,
    threads: int,
    progress_label: str = "",
) :
    """Corpus-fit hashed char TF-IDF; each sparse product retains only top k.

    The hash mapping is fixed, while IDF is fitted independently for each
    source/country/field corpus. Very common grams are removed from both sides.
    """
    if not target_texts:
        for _ in query_texts:
            yield []
        return
    index = fit_hashed_tfidf(
        target_texts, analyzer="char", ngram_range=ngram_range,
        n_features=n_features, max_document_frequency=max_document_frequency,
    )
    for start in range(0, len(query_texts), query_batch_size):
        if progress_label and start % (query_batch_size * 25) == 0:
            print(f"{progress_label}: {start:,}/{len(query_texts):,} queries", flush=True)
        batch = query_texts[start:start + query_batch_size]
        product = query_hashed_tfidf(index, batch, k=k, threads=threads)
        for row in range(product.shape[0]):
            left, right = product.indptr[row:row + 2]
            pairs = [(target_ids[int(col)], float(score))
                     for col, score in zip(product.indices[left:right], product.data[left:right])]
            pairs.sort(key=lambda item: (-item[1], item[0]))
            yield pairs
