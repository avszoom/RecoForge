"""Phase 5 tests — UserState, on_click, adaptive recommendations.

Covers:
    - UserStateStore round-trip
    - record_click caps history + invalidates session embedding
    - final_user_embedding falls back to long-term when no clicks
    - on_click flow updates state + trending counter + history
    - adaptive recommendations shift toward clicked categories
    - Phase 4 (long_term) mode unchanged
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.serving.recommender import Recommender
from src.serving.user_state import RECENT_ITEMS_CAP, UserState, UserStateStore


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"


def _required_paths_exist() -> bool:
    return all([
        (ARTIFACTS / "item_index.faiss").exists(),
        (ARTIFACTS / "item_embeddings.npy").exists(),
        (ARTIFACTS / "user_embeddings.npy").exists(),
        (DATA / "items.jsonl").exists(),
    ])


# ─── UserState / UserStateStore ──────────────────────────────────────────


def test_record_click_caps_history_and_invalidates_session() -> None:
    s = UserState(user_id="u_test")
    s.session_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(RECENT_ITEMS_CAP + 5):
        s.record_click(f"item_{i:03d}", "Travel")
    assert len(s.recent_clicked_items) == RECENT_ITEMS_CAP
    assert s.recent_clicked_items[-1] == f"item_{RECENT_ITEMS_CAP + 4:03d}"
    assert s.recent_clicked_items[0] == f"item_{5:03d}"             # oldest evicted
    assert s.session_embedding is None                              # invalidated by click
    assert s.recent_categories == Counter({"Travel": RECENT_ITEMS_CAP + 5})
    assert s.n_clicks_total == RECENT_ITEMS_CAP + 5


def test_userstatestore_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "user_state.json"
    store = UserStateStore(path)
    s = store.get_or_create("u_alpha")
    s.record_click("item_x", "Food")
    s.record_click("item_y", "Travel")
    store.save()

    reloaded = UserStateStore(path)
    s2 = reloaded.get("u_alpha")
    assert s2 is not None
    assert s2.recent_clicked_items == ["item_x", "item_y"]
    assert s2.recent_categories == Counter({"Food": 1, "Travel": 1})
    assert s2.n_clicks_total == 2


# ─── Recommender Phase-5 fixture (clean state per session) ───────────────


@pytest.fixture(scope="function")
def recommender(tmp_path: Path) -> Recommender:
    if not _required_paths_exist():
        pytest.skip("artifacts/ + data/ not fully populated — run the pipeline first")
    return Recommender(ARTIFACTS, DATA, user_state_path=tmp_path / "state.json")


# ─── final_user_embedding ────────────────────────────────────────────────


def test_final_user_embedding_no_clicks_equals_long_term(recommender: Recommender) -> None:
    user_id = next(iter(recommender.user_id_to_row))
    long_term = recommender.long_term_embedding(user_id)
    final = recommender.final_user_embedding(user_id)
    np.testing.assert_allclose(final, long_term, atol=1e-6)


def test_final_user_embedding_shifts_after_clicks(recommender: Recommender) -> None:
    user_id = next(iter(recommender.user_id_to_row))
    long_term = recommender.long_term_embedding(user_id).copy()

    # Pick three items from a category the user does NOT have in their interests
    # to maximize the embedding shift.
    profile = recommender.user_profile(user_id) or {}
    user_interests = set(profile.get("interests", []))
    foreign = next(
        cat for cat in recommender.items_by_category if cat not in user_interests
    )
    foreign_items = recommender.items_by_category[foreign][:3]

    for it in foreign_items:
        recommender.on_click(user_id, it["item_id"], persist=False)

    final = recommender.final_user_embedding(user_id)
    # The shift must be measurable but bounded (we still keep some long-term anchor).
    cos = float(np.dot(final, long_term))
    assert cos < 0.999, "final_user_embedding didn't move at all after 3 clicks"
    assert cos > 0.0, "final_user_embedding flipped to opposite of long-term — too aggressive"


# ─── on_click side effects ───────────────────────────────────────────────


def test_on_click_updates_state_history_and_trending(recommender: Recommender) -> None:
    user_id = next(iter(recommender.user_id_to_row))
    item_id = next(iter(recommender.items_by_id))
    pre_trending = recommender.trending_counter[item_id]
    pre_history = item_id in recommender.seen_items(user_id)

    recommender.on_click(user_id, item_id, persist=False)

    state = recommender.user_state.get(user_id)
    assert state is not None
    assert state.recent_clicked_items[-1] == item_id
    assert recommender.trending_counter[item_id] == pre_trending + 1.0
    assert item_id in recommender.seen_items(user_id)
    if not pre_history:
        # If the item wasn't in history before, on_click must have added it.
        assert item_id in recommender.user_history[user_id]


def test_on_click_unknown_user_raises(recommender: Recommender) -> None:
    item_id = next(iter(recommender.items_by_id))
    with pytest.raises(KeyError):
        recommender.on_click("u_DOES_NOT_EXIST", item_id, persist=False)


def test_on_click_unknown_item_raises(recommender: Recommender) -> None:
    user_id = next(iter(recommender.user_id_to_row))
    with pytest.raises(KeyError):
        recommender.on_click(user_id, "item_DOES_NOT_EXIST", persist=False)


# ─── adaptive recommendation behaviour ───────────────────────────────────


def test_adaptive_recs_shift_toward_clicked_category(recommender: Recommender) -> None:
    """The killer feature: clicking N items from a foreign category should make
    that category dominate the next adaptive recommendations."""
    # Pick a user with a single declared interest so the shift is visible.
    user_id = next(
        (
            uid for uid, u in recommender.users_by_id.items()
            if len(u["interests"]) == 1 and u["activity_level"] == "high"
        ),
        None,
    )
    if user_id is None:
        pytest.skip("no high-activity single-interest user in dataset")

    profile = recommender.user_profile(user_id)
    declared = profile["interests"][0]
    foreign = next(c for c in recommender.items_by_category if c != declared)
    foreign_items = recommender.items_by_category[foreign][:5]

    # Baseline (no clicks): top-10 should be dominated by the user's declared interest.
    baseline = recommender.recommend(user_id, k=10, mode="adaptive")
    baseline_declared = sum(1 for r in baseline if r.category == declared)

    # Click 5 items from the foreign category.
    for it in foreign_items:
        recommender.on_click(user_id, it["item_id"], persist=False)

    after = recommender.recommend(user_id, k=10, mode="adaptive")
    after_foreign = sum(1 for r in after if r.category == foreign)

    # Strong claim: at least half the post-click top-10 should be in the foreign category.
    assert after_foreign >= 5, (
        f"adaptive recs didn't shift after 5 clicks in {foreign}: "
        f"{after_foreign}/10 are {foreign}"
    )
    # And the declared-interest dominance should have weakened.
    after_declared = sum(1 for r in after if r.category == declared)
    assert after_declared < baseline_declared


def test_adaptive_results_have_source_tags(recommender: Recommender) -> None:
    user_id = next(iter(recommender.user_id_to_row))
    recs = recommender.recommend(user_id, k=10, mode="adaptive")
    assert len(recs) == 10
    valid_sources = {"ann", "recent", "trending", "fresh", "category"}
    for r in recs:
        assert r.sources, f"rec {r.item_id} has no source tags"
        assert all(s in valid_sources for s in r.sources)
        assert r.rank > 0
        assert 0.0 <= r.score <= 1.0


def test_long_term_mode_unchanged_after_phase5(recommender: Recommender) -> None:
    """Phase 4 behaviour is preserved when mode='long_term'."""
    user_id = next(
        uid for uid, u in recommender.users_by_id.items()
        if len(u["interests"]) == 1
    )
    interest = recommender.users_by_id[user_id]["interests"][0]
    recs = recommender.recommend(user_id, k=10, mode="long_term")
    in_interest = sum(1 for r in recs if r.category == interest)
    # mode='long_term' is the Phase 4 path: should be ≥8/10 on-category for single-interest users.
    assert in_interest >= 8
    # Single-source result.
    for r in recs:
        assert r.sources == ["ann"]
