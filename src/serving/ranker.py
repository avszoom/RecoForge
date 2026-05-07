"""Candidate merge → linear scoring → 80/10/10 diversification.

The scoring formula (from the design spec):
    score = 0.45 * ann
          + 0.25 * recent_similarity
          + 0.15 * category_match
          + 0.10 * trending
          + 0.05 * freshness

After ranking, the diversifier slots a top-10 page as:
    8 personalized   (top of the ranked list)
    1 trending       (highest-scoring trending candidate not already in)
    1 fresh          (highest-scoring fresh candidate not already in)

This delivers the design's "80% personalized / 10% trending / 10% fresh"
split while still respecting the linear ranker.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


# Source-weight contract (must align with generator names in candidate_generators.py).
RANK_WEIGHTS: dict[str, float] = {
    "ann":            0.45,
    "recent":         0.25,
    "category_match": 0.15,
    "trending":       0.10,
    "fresh":          0.05,
}

# Default top-k composition. Values must sum to k.
DEFAULT_K = 10
DEFAULT_PERSONALIZED_SLOTS = 8
DEFAULT_TRENDING_SLOTS = 1
DEFAULT_FRESH_SLOTS = 1


@dataclass
class MergedCandidate:
    item_id: str
    source_scores: dict[str, float] = field(default_factory=dict)   # source → its raw [0, 1] score
    final_score: float = 0.0
    title: str = ""
    category: str = ""
    topic: str = ""
    body: str = ""
    popularity: float = 0.0
    created_at: str = ""
    rank: int = 0

    @property
    def title_short(self) -> str:
        return (self.title[:65] + "…") if len(self.title) > 65 else self.title


# ─── merge ───────────────────────────────────────────────────────────────


def merge(candidates_per_source: list) -> dict[str, MergedCandidate]:
    """Dedupe across sources by item_id, keeping all per-source scores.

    If the same item appears multiple times within a single source (shouldn't
    happen but defensive), keep the maximum score.
    """
    merged: dict[str, MergedCandidate] = {}
    for source_list in candidates_per_source:
        for c in source_list:
            mc = merged.get(c.item_id)
            if mc is None:
                mc = MergedCandidate(item_id=c.item_id)
                merged[c.item_id] = mc
            mc.source_scores[c.source] = max(mc.source_scores.get(c.source, 0.0), c.source_score)
    return merged


# ─── score ───────────────────────────────────────────────────────────────


def rank(
    merged: dict[str, MergedCandidate],
    user_interests: set[str],
    items_by_id: dict[str, dict],
) -> list[MergedCandidate]:
    """Apply the linear scoring formula and return candidates sorted by final_score desc."""
    scored: list[MergedCandidate] = []
    for mc in merged.values():
        item = items_by_id.get(mc.item_id)
        if item is None:
            continue
        s = mc.source_scores
        cat_match = 1.0 if item["category"] in user_interests else 0.0
        mc.final_score = (
            RANK_WEIGHTS["ann"]            * s.get("ann", 0.0)
            + RANK_WEIGHTS["recent"]       * s.get("recent", 0.0)
            + RANK_WEIGHTS["category_match"] * cat_match
            + RANK_WEIGHTS["trending"]     * s.get("trending", 0.0)
            + RANK_WEIGHTS["fresh"]        * s.get("fresh", 0.0)
        )
        mc.title = item["title"]
        mc.category = item["category"]
        mc.topic = item["topic"]
        mc.body = item["body"]
        mc.popularity = float(item["popularity_score"])
        mc.created_at = item["created_at"]
        scored.append(mc)
    scored.sort(key=lambda c: c.final_score, reverse=True)
    return scored


# ─── re-rank: 80/10/10 split ─────────────────────────────────────────────


def diversify(
    ranked: list[MergedCandidate],
    seen_items: set[str],
    *,
    k: int = DEFAULT_K,
    personalized_slots: int = DEFAULT_PERSONALIZED_SLOTS,
    trending_slots: int = DEFAULT_TRENDING_SLOTS,
    fresh_slots: int = DEFAULT_FRESH_SLOTS,
) -> list[MergedCandidate]:
    """Compose the final top-k page: personalized + trending + fresh.

    `seen_items` is filtered out at every step. We use an ordered dict
    keyed by item_id to avoid double-counting and preserve insertion order.
    """
    out: "OrderedDict[str, MergedCandidate]" = OrderedDict()

    # 1. Personalized slots: top of the ranked list (any source).
    for mc in ranked:
        if len(out) >= personalized_slots:
            break
        if mc.item_id in seen_items or mc.item_id in out:
            continue
        out[mc.item_id] = mc

    # 2. Trending slot: best trending candidate not yet in the page.
    for slot in range(trending_slots):
        for mc in ranked:
            if mc.item_id in seen_items or mc.item_id in out:
                continue
            if "trending" in mc.source_scores:
                out[mc.item_id] = mc
                break

    # 3. Fresh slot: best fresh candidate not yet in the page.
    for slot in range(fresh_slots):
        for mc in ranked:
            if mc.item_id in seen_items or mc.item_id in out:
                continue
            if "fresh" in mc.source_scores:
                out[mc.item_id] = mc
                break

    # 4. Backfill: if any of the above buckets came up empty (no trending
    #    candidates available, etc.), fill from the ranked list.
    for mc in ranked:
        if len(out) >= k:
            break
        if mc.item_id in seen_items or mc.item_id in out:
            continue
        out[mc.item_id] = mc

    final = list(out.values())[:k]
    for i, mc in enumerate(final, start=1):
        mc.rank = i
    return final
