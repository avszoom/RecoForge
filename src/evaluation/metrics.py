"""Pure ranking-evaluation metrics.

All functions take a list of `(true_item_id, ranked_item_ids)` pairs — the
ground-truth item the user actually engaged with, and the ranked list the
recommender produced — and return a scalar mean across the batch.

Single-relevant-item assumption: each query has exactly one ground-truth
item (the next interaction in the eval log). For multi-label or graded
relevance you'd need a different formulation; we don't need that here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _rank_of(true: str, ranked: Sequence[str], k: int) -> int | None:
    """Return 1-indexed rank of `true` in `ranked[:k]`, or None if absent."""
    try:
        return ranked[:k].index(true) + 1
    except ValueError:
        return None


def recall_at_k(pairs: list[tuple[str, Sequence[str]]], k: int) -> float:
    """Fraction of queries where the true item appears in the top-k."""
    if not pairs:
        return 0.0
    hits = sum(1 for true, ranked in pairs if _rank_of(true, ranked, k) is not None)
    return hits / len(pairs)


def precision_at_k(pairs: list[tuple[str, Sequence[str]]], k: int) -> float:
    """Same numerator as recall@k, divided by k. With 1 ground-truth per query
    this equals recall@k / k — included for completeness."""
    if not pairs or k <= 0:
        return 0.0
    hits = sum(1 for true, ranked in pairs if _rank_of(true, ranked, k) is not None)
    return hits / (len(pairs) * k)


def mrr(pairs: list[tuple[str, Sequence[str]]], k: int) -> float:
    """Mean reciprocal rank, capped at k. 1/rank if true is in top-k, else 0."""
    if not pairs:
        return 0.0
    total = 0.0
    for true, ranked in pairs:
        r = _rank_of(true, ranked, k)
        if r is not None:
            total += 1.0 / r
    return total / len(pairs)


def ndcg_at_k(pairs: list[tuple[str, Sequence[str]]], k: int) -> float:
    """NDCG@k for a single ground-truth-per-query setting.

    DCG = 1/log2(rank+1) if true is in top-k, else 0.
    IDCG = 1/log2(2) = 1.0 (best case: rank 1).
    So NDCG@k = DCG (the IDCG normalizer collapses to 1).
    """
    if not pairs:
        return 0.0
    total = 0.0
    for true, ranked in pairs:
        r = _rank_of(true, ranked, k)
        if r is not None:
            total += 1.0 / math.log2(r + 1)
    return total / len(pairs)


def category_match_at_1(
    pairs: list[tuple[str, Sequence[str]]],
    item_to_category: dict[str, str],
    user_interests_by_query: list[set[str]],
) -> float:
    """Of the top-1 items, fraction that fall in the user's declared interests.

    Useful as a sanity floor: even if exact-item recall is low, this should
    be high if the model has learned categories correctly.
    """
    if not pairs:
        return 0.0
    n = 0
    matches = 0
    for (_, ranked), interests in zip(pairs, user_interests_by_query):
        if not ranked or not interests:
            continue
        cat = item_to_category.get(ranked[0])
        if cat is None:
            continue
        n += 1
        if cat in interests:
            matches += 1
    return matches / max(1, n)
