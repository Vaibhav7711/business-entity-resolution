"""Candidate routes added in Phase 1B; all vocabulary comes from training data.

The legal suffix spellings are the six variants printed in the supplied
problem statement (student_resource/README.md, Name variations).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Iterable, Iterator

from .blocking import HashedTfidfIndex, fit_hashed_tfidf, query_hashed_tfidf


LEGAL_SUFFIXES = frozenset({"corp", "corporation", "pvt", "private", "ltd", "limited"})
DIGIT_PATTERN = re.compile(r"\d+")
WORD_TOKEN_PATTERN = r"(?u)\b\w+\b"


def strip_legal_suffix(name: str) -> str:
    """Remove up to two trailing statement-listed suffix tokens, if safe."""
    tokens = name.split()
    removed = 0
    while len(tokens) > 1 and tokens[-1] in LEGAL_SUFFIXES and removed < 2:
        tokens.pop()
        removed += 1
    return " ".join(tokens)


def name_tokens(value: str) -> set[str]:
    return {token for token in value.split() if len(token) >= 2}


def address_digits(value: str) -> set[str]:
    return {token for token in DIGIT_PATTERN.findall(value) if len(token) >= 2}


class RareTokenIndex:
    """Frequency-capped inverted index over one complete target partition.

    Target token frequencies are counted from the supplied partition itself,
    restricted to ``wanted`` query tokens. Tokens above the frequency cap cannot
    enter the index. A query's result depends only on that query and the target
    partition, provided ``wanted`` contains all of the query's tokens.
    """

    def __init__(
        self, target_texts: list[str], target_ids: list[int], *,
        tokenize: Callable[[str], set[str]], wanted: set[str], max_document_frequency: int,
    ) -> None:
        self.tokenize = tokenize
        frequency: Counter[str] = Counter()
        for text in target_texts:
            frequency.update(tokenize(text) & wanted)
        self.frequency = frequency
        self.usable = {token for token in wanted if 0 < frequency[token] <= max_document_frequency}
        postings: dict[str, list[int]] = defaultdict(list)
        for text, target_id in zip(target_texts, target_ids):
            for token in tokenize(text) & self.usable:
                postings[token].append(target_id)
        self.postings = postings

    def query(self, text: str, *, max_query_tokens: int, k: int) -> list[tuple[int, float]]:
        """Rarest query tokens; hits ordered by summed inverse frequency, then ID."""
        frequency = self.frequency
        chosen = sorted(self.tokenize(text) & self.usable,
                        key=lambda token: (frequency[token], token))[:max_query_tokens]
        scores: dict[int, float] = defaultdict(float)
        for token in chosen:
            weight = 1.0 / math.log2(2 + frequency[token])
            for target_id in self.postings[token]:
                scores[target_id] += weight
        ranked = sorted(scores, key=lambda value: (-scores[value], value))[:k]
        return [(value, scores[value]) for value in ranked]


def rare_token_lookup(
    query_texts: list[str], target_texts: list[str], target_ids: list[int],
    *, tokenize: Callable[[str], set[str]], max_document_frequency: int,
    max_query_tokens: int, k: int,
) -> list[list[int]]:
    """Lookup the rarest query tokens against one complete target partition."""
    query_tokens = [tokenize(text) for text in query_texts]
    wanted = set().union(*query_tokens) if query_tokens else set()
    index = RareTokenIndex(target_texts, target_ids, tokenize=tokenize, wanted=wanted,
                           max_document_frequency=max_document_frequency)
    return [[value for value, _ in index.query(text, max_query_tokens=max_query_tokens, k=k)]
            for text in query_texts]


def fit_word_tfidf(
    target_names: list[str], *, n_features: int, max_document_frequency: float,
) -> HashedTfidfIndex:
    """Word unigram/bigram hashed TF-IDF index for one target partition."""
    return fit_hashed_tfidf(
        target_names, analyzer="word", token_pattern=WORD_TOKEN_PATTERN, ngram_range=(1, 2),
        n_features=n_features, max_document_frequency=max_document_frequency,
    )


def word_tfidf_topk_rows(
    query_names: list[str], target_names: list[str], target_ids: list[int],
    *, k: int, n_features: int, max_document_frequency: float,
    query_batch_size: int, threads: int,
) -> Iterator[list[tuple[int, float]]]:
    """Word unigram/bigram hashed TF-IDF search with bounded sparse top-k."""
    index = fit_word_tfidf(target_names, n_features=n_features,
                           max_document_frequency=max_document_frequency)
    for start in range(0, len(query_names), query_batch_size):
        if start % (25 * query_batch_size) == 0:
            print(f"word_tfidf: {start:,}/{len(query_names):,} queries", flush=True)
        product = query_hashed_tfidf(index, query_names[start:start + query_batch_size],
                                     k=k, threads=threads)
        for row in range(product.shape[0]):
            left, right = product.indptr[row:row + 2]
            pairs = [(target_ids[int(col)], float(score))
                     for col, score in zip(product.indices[left:right], product.data[left:right])]
            yield sorted(pairs, key=lambda item: (-item[1], item[0]))


def stripped_exact_lookup(
    query_names: list[str], target_names: list[str], target_ids: list[int],
) -> list[list[int]]:
    """Exact lookup on names with only the stated terminal legal forms removed."""
    query_keys = [strip_legal_suffix(name) for name in query_names]
    wanted = set(query_keys) - {""}
    index: dict[str, list[int]] = defaultdict(list)
    for name, target_id in zip(target_names, target_ids):
        key = strip_legal_suffix(name)
        if key in wanted:
            index[key].append(target_id)
    return [index.get(key, []).copy() if key else [] for key in query_keys]


def build_key_index(
    target_keys: Iterable[str], target_ids: list[int], wanted: set[str],
) -> dict[str, list[int]]:
    """All target IDs per wanted non-empty key, in target file order.

    Used for both exact normalized-name and suffix-stripped-name lookup; no
    target is dropped because another target shares its text.
    """
    wanted = wanted - {""}
    index: dict[str, list[int]] = defaultdict(list)
    for key, target_id in zip(target_keys, target_ids):
        if key in wanted:
            index[key].append(target_id)
    return index
