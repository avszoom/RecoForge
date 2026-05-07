"""Multi-source candidate generators for the adaptive recommender.

Each generator returns a flat list of `Candidate` records tagged with a
`source` so the merger can keep track of which generator contributed
which item. All scores are normalized into [0, 1] within each source so
the linear ranker can combine them on a comparable scale.

Generators (per the design spec):
    1. ann_long_term       — FAISS over the blended user embedding
    2. similar_to_recent   — FAISS over each recently clicked item
    3. trending            — top items by recent click count
    4. fresh               — newest items (created in the last 7 days)
    5. category_interest   — items from the user's preferred categories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                # pragma: no cover
    from src.serving.recommender import Recommender


@dataclass
class Candidate:
    item_id: str
    source: str            # "ann" | "recent" | "trending" | "fresh" | "category"
    source_score: float    # normalized to [0, 1]


# ─── helpers ─────────────────────────────────────────────────────────────


def _cosine_to_unit(score: float) -> float:
    """Map cosine similarity from [-1, 1] to [0, 1]."""
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


# ─── generators ──────────────────────────────────────────────────────────


def ann_long_term(
    rec: "Recommender", user_id: str,
    k: int = 50, *,
    blend: tuple[float, float] | None = None,
) -> list[Candidate]:
    """Top-k from FAISS using the (blended) user embedding.

    This is the workhorse — most candidates come from here. Uses
    `final_user_embedding`, which already incorporates the session blend
    when the user has recent clicks. `blend` overrides the auto schedule.
    """
    user_emb = rec.final_user_embedding(user_id, blend=blend)
    hits = rec.index.search(user_emb, k=k)[0]
    return [Candidate(iid, "ann", _cosine_to_unit(score)) for iid, score in hits]


def similar_to_recent(
    rec: "Recommender", user_id: str,
    *,
    n_recent: int = 5,
    k_per_item: int = 10,
    total_cap: int = 50,
) -> list[Candidate]:
    """For each of the user's last N clicks, find FAISS neighbors.

    For each candidate item, keep the MAX similarity across all recent
    clicks — i.e. "how close is this item to the most-similar thing the
    user just engaged with". Items the user just clicked are filtered out.
    """
    state = rec.user_state.get_or_create(user_id)
    if not state.recent_clicked_items:
        return []

    recent = state.recent_clicked_items[-n_recent:]
    recent_set = set(recent)

    rows = [rec.index.item_id_to_row[iid] for iid in recent if iid in rec.index.item_id_to_row]
    if not rows:
        return []
    queries = rec.item_emb[rows]               # (n_recent, dim)

    aggregated: dict[str, float] = {}
    hits_per_query = rec.index.search(queries, k=k_per_item)
    for hits in hits_per_query:
        for iid, score in hits:
            if iid in recent_set:
                continue
            normed = _cosine_to_unit(score)
            if normed > aggregated.get(iid, 0.0):
                aggregated[iid] = normed

    sorted_items = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)[:total_cap]
    return [Candidate(iid, "recent", s) for iid, s in sorted_items]


def trending(rec: "Recommender", k: int = 30) -> list[Candidate]:
    """Top items by click count, normalized to [0, 1] using the max.

    The Recommender maintains a Counter that's seeded from the offline
    click log at startup and incremented on every `on_click` at runtime.
    """
    counter = rec.trending_counter
    if not counter:
        return []
    top = counter.most_common(k)
    max_count = max(top[0][1], 1)
    return [Candidate(iid, "trending", count / max_count) for iid, count in top]


def fresh(rec: "Recommender", k: int = 30) -> list[Candidate]:
    """Newest items by created_at — score = 1.0 brand-new, decays linearly over 7 days."""
    if not rec.fresh_items_sorted:
        return []
    head = rec.fresh_items_sorted[:k]
    return [
        Candidate(iid, "fresh", max(0.0, min(1.0, 1.0 - age_days / 7.0)))
        for iid, age_days in head
    ]


def category_interest(rec: "Recommender", user_id: str, k: int = 30) -> list[Candidate]:
    """Items from the user's preferred categories, ranked by popularity_score.

    Acts as a robust fallback — even if every other generator fails (or
    returns nothing for a brand-new user with no clicks), this still
    surfaces relevant items as long as the user declared any interests.
    """
    profile = rec.user_profile(user_id)
    if not profile:
        return []
    interests = profile.get("interests") or []
    if not interests:
        return []

    pool: list[tuple[str, float]] = []
    for cat in interests:
        for item in rec.items_by_category.get(cat, []):
            pool.append((item["item_id"], float(item["popularity_score"])))
    if not pool:
        return []
    pool.sort(key=lambda x: x[1], reverse=True)
    head = pool[:k]
    max_pop = max(head[0][1], 1e-6)
    return [Candidate(iid, "category", pop / max_pop) for iid, pop in head]


# ─── one-call orchestrator ───────────────────────────────────────────────


def generate_all(
    rec: "Recommender", user_id: str, *,
    blend: tuple[float, float] | None = None,
) -> list[list[Candidate]]:
    """Run every generator. `blend` overrides the auto blend schedule for ann_long_term."""
    return [
        ann_long_term(rec, user_id, k=50, blend=blend),
        similar_to_recent(rec, user_id),
        trending(rec, k=30),
        fresh(rec, k=30),
        category_interest(rec, user_id, k=30),
    ]
