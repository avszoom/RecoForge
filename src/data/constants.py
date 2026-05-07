"""Canonical taxonomy used by users, items, and the interaction generator.

Single source of truth so we never have a category that exists for users but
not for items, or vice versa. Imported by every module that needs to reason
about categories.
"""

from __future__ import annotations

# 12 categories. Kept small for POC — large enough to show topical clustering,
# small enough that a 1k-user / 4k-item dataset has real signal per category.
CATEGORIES: tuple[str, ...] = (
    "AI Infrastructure",
    "Startups",
    "Investing",
    "Travel",
    "Food",
    "Health & Fitness",
    "Programming",
    "Personal Finance",
    "Science",
    "Gaming",
    "Music",
    "Movies & TV",
)

# Hand-curated adjacency: which categories are "related" enough that a user
# interested in A might plausibly engage with content from B. Used by the
# interaction generator (20% of clicks come from related categories).
RELATED: dict[str, tuple[str, ...]] = {
    "AI Infrastructure":  ("Programming", "Startups", "Science"),
    "Startups":           ("Investing", "AI Infrastructure", "Personal Finance"),
    "Investing":          ("Personal Finance", "Startups"),
    "Travel":             ("Food", "Movies & TV"),
    "Food":               ("Travel", "Health & Fitness"),
    "Health & Fitness":   ("Food", "Science"),
    "Programming":        ("AI Infrastructure", "Gaming"),
    "Personal Finance":   ("Investing", "Startups"),
    "Science":            ("AI Infrastructure", "Health & Fitness"),
    "Gaming":             ("Programming", "Movies & TV", "Music"),
    "Music":              ("Movies & TV", "Gaming"),
    "Movies & TV":        ("Music", "Gaming", "Travel"),
}

ACTIVITY_LEVELS: tuple[str, ...] = ("low", "medium", "high")

# Mean number of interactions per user given activity level.
# Used as the lambda for a Poisson draw in the interaction generator.
ACTIVITY_TO_INTERACTION_MEAN: dict[str, int] = {
    "low":    30,
    "medium": 80,
    "high":  200,
}

AGE_BUCKETS: tuple[str, ...] = ("18-24", "25-34", "35-44", "45-54", "55+")
LOCATIONS: tuple[str, ...] = ("US", "EU", "UK", "IN", "APAC", "LATAM")

# Event types and their training weights (as in the design spec).
EVENT_WEIGHTS: dict[str, float] = {
    "view":  0.2,
    "click": 1.0,
    "like":  2.0,
    "save":  3.0,
    "share": 4.0,
}

# Empirical-ish distribution of events given that the user engaged at all.
# Sums to 1.0. Most engagements are clicks; shares are rare but high-signal.
EVENT_TYPE_DISTRIBUTION: dict[str, float] = {
    "view":  0.45,
    "click": 0.35,
    "like":  0.12,
    "save":  0.05,
    "share": 0.03,
}

# How interaction sources are mixed for each user (must sum to 1.0).
SOURCE_MIX: dict[str, float] = {
    "preferred": 0.70,
    "related":   0.20,
    "random":    0.10,
}


def assert_constants_consistent() -> None:
    """Cheap invariants — fail loudly if the taxonomy gets out of sync."""
    assert set(RELATED.keys()) == set(CATEGORIES), "RELATED keys must match CATEGORIES"
    for cat, rels in RELATED.items():
        for r in rels:
            assert r in CATEGORIES, f"RELATED[{cat}] contains unknown category {r!r}"
    assert abs(sum(EVENT_TYPE_DISTRIBUTION.values()) - 1.0) < 1e-6
    assert abs(sum(SOURCE_MIX.values()) - 1.0) < 1e-6
    assert set(ACTIVITY_TO_INTERACTION_MEAN.keys()) == set(ACTIVITY_LEVELS)


assert_constants_consistent()
