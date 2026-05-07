"""Phase 4 — long-term-only recommender tests.

These tests assume the full pipeline has already been run:
    python -m src.data.generate_dataset
    python -m src.models.text_features
    python -m src.models.train_two_tower
    python -m src.models.export_embeddings
    python -m src.indexing.build_faiss

If artifacts/ or data/ aren't populated, the tests skip rather than fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.serving.recommender import Recommendation, Recommender


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"


def _required_paths_exist() -> bool:
    return all([
        (ARTIFACTS / "item_index.faiss").exists(),
        (ARTIFACTS / "user_embeddings.npy").exists(),
        (ARTIFACTS / "user_id_to_row.json").exists(),
        (DATA / "items.jsonl").exists(),
        (DATA / "users.jsonl").exists(),
    ])


@pytest.fixture(scope="module")
def recommender() -> Recommender:
    if not _required_paths_exist():
        pytest.skip("artifacts/ + data/ not fully populated — run the pipeline first")
    return Recommender(ARTIFACTS, DATA)


def test_loads_with_expected_shapes(recommender: Recommender) -> None:
    assert recommender.user_emb.shape[0] == len(recommender.user_id_to_row)
    assert recommender.user_emb.shape[1] == recommender.index.dim
    assert recommender.index.n_items == len(recommender.items_by_id)


def test_recommend_returns_k_items(recommender: Recommender) -> None:
    user_id = next(iter(recommender.user_id_to_row))
    recs = recommender.recommend(user_id, k=10)
    assert len(recs) == 10
    assert all(isinstance(r, Recommendation) for r in recs)
    assert all(r.rank == i + 1 for i, r in enumerate(recs))
    # cosine scores are L2-normalized dot products → must be in [-1, 1]
    assert all(-1.0 <= r.score <= 1.0001 for r in recs)
    # results should be sorted by score descending
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_filter_seen_actually_filters(recommender: Recommender) -> None:
    """The basic contract: items in the user's history never appear when filter_seen=True.

    We also assert the behavioral contract by injecting a known top-ranked item
    into the user's history and verifying it disappears from the next filtered call.
    Avoids relying on the user's natural history happening to overlap their top-N.
    """
    user_id = next(
        (uid for uid in recommender.user_id_to_row if recommender.seen_items(uid)),
        None,
    )
    if user_id is None:
        pytest.skip("no users with logged history")

    seen = recommender.seen_items(user_id)

    # Contract 1: nothing in `seen` shows up when filter_seen=True.
    recs = recommender.recommend(user_id, k=20, filter_seen=True)
    assert all(r.item_id not in seen for r in recs)

    # Contract 2: artificially seeing the current top-1 must remove it from the next
    # filtered call. This proves the filter is actually consulted, not just decorative.
    top_unfiltered = recommender.recommend(user_id, k=1, filter_seen=False)[0]
    original = recommender.user_history.get(user_id, set()).copy()
    try:
        recommender.user_history[user_id] = original | {top_unfiltered.item_id}
        recs_after = recommender.recommend(user_id, k=20, filter_seen=True)
        assert top_unfiltered.item_id not in {r.item_id for r in recs_after}
    finally:
        recommender.user_history[user_id] = original


def test_unknown_user_raises(recommender: Recommender) -> None:
    with pytest.raises(KeyError, match="unknown user_id"):
        recommender.recommend("u_DOES_NOT_EXIST", k=5)


def test_category_match_for_single_interest_user(recommender: Recommender) -> None:
    """For users with a single declared interest, top-10 should be ≥ 80% on-category.

    With cat_match@1 = 100% from the Phase 2 retrieval probe, top-10 should be
    overwhelmingly dominated by the user's interest. We allow a small slack
    because the seen-item filter occasionally leaks in a related-category item.
    """
    matches = 0
    n_users_checked = 0
    for u in recommender.users_by_id.values():
        if len(u["interests"]) != 1:
            continue
        recs = recommender.recommend(u["user_id"], k=10)
        if not recs:
            continue
        in_interest = sum(1 for r in recs if r.category == u["interests"][0])
        if in_interest >= 8:           # ≥ 80% threshold, generous
            matches += 1
        n_users_checked += 1
        if n_users_checked >= 20:      # 20 single-interest users is enough
            break
    assert n_users_checked > 0
    pass_rate = matches / n_users_checked
    assert pass_rate >= 0.85, f"only {matches}/{n_users_checked} single-interest users got ≥80% category-match"
