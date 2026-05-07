"""Tests for src/evaluation/metrics.py.

Pure functions — no dataset dependency. Run independently of the full pipeline.
"""

from __future__ import annotations

import math

import pytest

from src.evaluation.metrics import (
    category_match_at_1,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ─── recall@k ────────────────────────────────────────────────────────────


def test_recall_at_k_perfect() -> None:
    pairs = [("a", ["a", "b", "c"]), ("b", ["b", "c", "a"])]
    assert recall_at_k(pairs, k=1) == 1.0
    assert recall_at_k(pairs, k=10) == 1.0


def test_recall_at_k_zero() -> None:
    pairs = [("a", ["x", "y", "z"]), ("b", ["x", "y", "z"])]
    assert recall_at_k(pairs, k=10) == 0.0


def test_recall_at_k_partial() -> None:
    pairs = [
        ("a", ["a", "x", "y"]),    # rank 1, hit
        ("b", ["x", "b", "y"]),    # rank 2, hit at k>=2
        ("c", ["x", "y", "z"]),    # not in list
    ]
    assert recall_at_k(pairs, k=1) == pytest.approx(1 / 3)   # only "a" at rank 1
    assert recall_at_k(pairs, k=2) == pytest.approx(2 / 3)
    assert recall_at_k(pairs, k=10) == pytest.approx(2 / 3)


def test_recall_at_k_empty_pairs() -> None:
    assert recall_at_k([], k=10) == 0.0


def test_recall_at_k_empty_ranking() -> None:
    pairs = [("a", [])]
    assert recall_at_k(pairs, k=10) == 0.0


# ─── precision@k ─────────────────────────────────────────────────────────


def test_precision_at_k_basic() -> None:
    pairs = [("a", ["a", "x", "y"]), ("b", ["b", "x", "y"])]
    # 2/2 hits at k=10, divided by k=10 → 0.1 per query, average = 0.1
    assert precision_at_k(pairs, k=10) == pytest.approx(0.1)


def test_precision_at_k_zero_k() -> None:
    pairs = [("a", ["a"])]
    assert precision_at_k(pairs, k=0) == 0.0


# ─── MRR ─────────────────────────────────────────────────────────────────


def test_mrr_known_values() -> None:
    pairs = [
        ("a", ["a", "x", "y"]),    # rank 1 → 1.0
        ("b", ["x", "b", "y"]),    # rank 2 → 0.5
        ("c", ["x", "y", "c"]),    # rank 3 → 1/3
        ("d", ["x", "y", "z"]),    # not in top-k → 0
    ]
    expected = (1.0 + 0.5 + 1 / 3 + 0.0) / 4
    assert mrr(pairs, k=10) == pytest.approx(expected)


def test_mrr_caps_at_k() -> None:
    pairs = [("a", ["x", "y", "a"])]   # rank 3
    assert mrr(pairs, k=2) == 0.0      # outside top-2
    assert mrr(pairs, k=3) == pytest.approx(1 / 3)


def test_mrr_empty() -> None:
    assert mrr([], k=10) == 0.0


# ─── NDCG@k ──────────────────────────────────────────────────────────────


def test_ndcg_known_values() -> None:
    pairs = [
        ("a", ["a", "x"]),    # rank 1 → 1/log2(2) = 1.0
        ("b", ["x", "b"]),    # rank 2 → 1/log2(3)
        ("c", ["x", "y"]),    # not in top-k → 0
    ]
    expected = (1.0 + (1.0 / math.log2(3)) + 0.0) / 3
    assert ndcg_at_k(pairs, k=10) == pytest.approx(expected)


def test_ndcg_decreasing_with_rank() -> None:
    """A hit at a lower rank should score less than at a higher rank."""
    high = [("a", ["a", "x", "y", "z"])]    # rank 1
    low = [("a", ["x", "y", "z", "a"])]     # rank 4
    assert ndcg_at_k(high, k=10) > ndcg_at_k(low, k=10)


def test_ndcg_empty() -> None:
    assert ndcg_at_k([], k=10) == 0.0


# ─── category_match@1 ────────────────────────────────────────────────────


def test_category_match_at_1() -> None:
    item_to_cat = {"a": "Travel", "b": "Food", "c": "Travel"}
    pairs = [
        ("a", ["a", "b"]),   # top-1 = a (Travel)
        ("a", ["c", "b"]),   # top-1 = c (Travel)
        ("b", ["b", "a"]),   # top-1 = b (Food)
    ]
    user_interests = [{"Travel"}, {"Travel"}, {"Travel"}]   # 3rd user wants Travel, gets Food
    rate = category_match_at_1(pairs, item_to_cat, user_interests)
    assert rate == pytest.approx(2 / 3)


def test_category_match_skips_empty_lists() -> None:
    item_to_cat = {"a": "Travel"}
    pairs = [("a", []), ("a", ["a"])]
    user_interests = [{"Travel"}, {"Travel"}]
    # First pair skipped (empty ranking), second one matches
    assert category_match_at_1(pairs, item_to_cat, user_interests) == 1.0
