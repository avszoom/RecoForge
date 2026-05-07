"""Schema + referential-integrity tests on the generated dataset.

These assume `python -m src.data.generate_dataset` has already been run
(default output path: ./data/). They are cheap and catch the typical
regressions: drifted schema, dangling foreign keys, broken taxonomy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.constants import (
    ACTIVITY_LEVELS,
    AGE_BUCKETS,
    CATEGORIES,
    EVENT_WEIGHTS,
    LOCATIONS,
)


DATA = Path(__file__).resolve().parent.parent / "data"

REQUIRED_USER_FIELDS = {"user_id", "age_bucket", "location", "interests", "activity_level"}
REQUIRED_ITEM_FIELDS = {
    "item_id", "category", "topic", "title", "body",
    "creator_id", "created_at", "popularity_score",
}
REQUIRED_INTERACTION_FIELDS = {
    "user_id", "item_id", "event_type", "event_weight", "timestamp",
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path} not present — run `python -m src.data.generate_dataset` first")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def users() -> list[dict]:
    return _load_jsonl(DATA / "users.jsonl")


@pytest.fixture(scope="module")
def items() -> list[dict]:
    return _load_jsonl(DATA / "items.jsonl")


@pytest.fixture(scope="module")
def interactions() -> list[dict]:
    return _load_jsonl(DATA / "interactions.jsonl")


def test_users_schema(users: list[dict]) -> None:
    assert len(users) > 0
    user_ids = set()
    for u in users:
        assert REQUIRED_USER_FIELDS.issubset(u.keys()), f"missing fields in {u}"
        assert u["age_bucket"] in AGE_BUCKETS
        assert u["location"] in LOCATIONS
        assert u["activity_level"] in ACTIVITY_LEVELS
        assert isinstance(u["interests"], list) and 1 <= len(u["interests"]) <= 3
        for c in u["interests"]:
            assert c in CATEGORIES, f"unknown interest {c}"
        user_ids.add(u["user_id"])
    assert len(user_ids) == len(users), "duplicate user_id"


def test_items_schema(items: list[dict]) -> None:
    assert len(items) > 0
    item_ids = set()
    for it in items:
        assert REQUIRED_ITEM_FIELDS.issubset(it.keys()), f"missing fields in {it}"
        assert it["category"] in CATEGORIES
        assert isinstance(it["title"], str) and len(it["title"]) > 0
        assert isinstance(it["body"], str) and len(it["body"]) > 0
        assert 0.0 <= it["popularity_score"] <= 1.0
        item_ids.add(it["item_id"])
    assert len(item_ids) == len(items), "duplicate item_id"


def test_interactions_schema_and_refs(
    users: list[dict], items: list[dict], interactions: list[dict]
) -> None:
    user_ids = {u["user_id"] for u in users}
    item_ids = {it["item_id"] for it in items}

    assert len(interactions) > 0
    for i in interactions:
        assert REQUIRED_INTERACTION_FIELDS.issubset(i.keys()), f"missing fields in {i}"
        assert i["user_id"] in user_ids, f"dangling user_id {i['user_id']}"
        assert i["item_id"] in item_ids, f"dangling item_id {i['item_id']}"
        assert i["event_type"] in EVENT_WEIGHTS
        assert i["event_weight"] == EVENT_WEIGHTS[i["event_type"]]


def test_preferred_category_mix(
    users: list[dict], items: list[dict], interactions: list[dict]
) -> None:
    """Sanity: ≥ 60% of interactions land in the user's declared interests.

    The generator targets 70%; we leave headroom for sampling variance.
    """
    user_interests = {u["user_id"]: set(u["interests"]) for u in users}
    item_cat = {it["item_id"]: it["category"] for it in items}
    hits = sum(
        1 for i in interactions
        if item_cat[i["item_id"]] in user_interests[i["user_id"]]
    )
    rate = hits / len(interactions)
    assert rate >= 0.60, f"preferred-category hit rate too low: {rate:.3f}"
