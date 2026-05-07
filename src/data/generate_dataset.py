"""Synthetic dataset generator for RecoForge.

Produces three JSONL files in `data/`:
    - users.jsonl              (1000 user profiles)
    - items.jsonl              (4000 items with title/body/metadata)
    - interactions.jsonl       (training interactions, last ~30 days)
    - interactions_eval.jsonl  (last 7 days, held out for offline evaluation)

Usage:
    python -m src.data.generate_dataset                 # defaults
    python -m src.data.generate_dataset --seed 7
    python -m src.data.generate_dataset --users 500 --items 2000 --out data/

Generation rules (from the design spec):
    - 70% of a user's interactions come from their preferred categories
    - 20% from related categories
    - 10% random exploration
    - Interaction count per user is Poisson(activity_level_mean)
    - Item popularity follows a power-law (a few hot items, long tail)
    - Last 7 days of interactions are held out as evaluation set
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data.constants import (
    ACTIVITY_LEVELS,
    ACTIVITY_TO_INTERACTION_MEAN,
    AGE_BUCKETS,
    CATEGORIES,
    EVENT_TYPE_DISTRIBUTION,
    EVENT_WEIGHTS,
    LOCATIONS,
    RELATED,
    SOURCE_MIX,
)
from src.data.content_templates import (
    CATEGORY_CONTENT,
    MODIFIERS,
    assert_content_complete,
)


# ─── logging ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_dataset")


# ─── records ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class User:
    user_id: str
    age_bucket: str
    location: str
    interests: list[str]
    activity_level: str


@dataclass(frozen=True)
class Item:
    item_id: str
    category: str
    topic: str
    title: str
    body: str
    creator_id: str
    created_at: str
    popularity_score: float


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    event_type: str
    event_weight: float
    timestamp: str


# ─── generation ──────────────────────────────────────────────────────────


def generate_users(n: int, rng: random.Random) -> list[User]:
    """Build n user profiles. Interests sampled from CATEGORIES (1–3 each)."""
    activity_weights = [0.4, 0.4, 0.2]   # 40% low, 40% medium, 20% high
    users: list[User] = []
    for i in range(n):
        n_interests = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)[0]
        interests = rng.sample(CATEGORIES, k=n_interests)
        users.append(
            User(
                user_id=f"u_{i:04d}",
                age_bucket=rng.choice(AGE_BUCKETS),
                location=rng.choice(LOCATIONS),
                interests=list(interests),
                activity_level=rng.choices(ACTIVITY_LEVELS, weights=activity_weights, k=1)[0],
            )
        )
    return users


def _power_law_popularity(rng: random.Random) -> float:
    """Heavy-tailed popularity score in [0, 1].

    A few items are very popular (≈0.8+), most are middling, a long tail
    is near zero. This shapes which items the interaction generator
    favors when sampling by category.
    """
    # Pareto draw, capped to keep the score in a reasonable range.
    raw = rng.paretovariate(2.0) - 1.0     # ≥ 0, mean ≈ 1
    return max(0.0, min(1.0, raw / 5.0))


def generate_items(n: int, rng: random.Random, *, n_creators: int = 200) -> list[Item]:
    """Build n items with templated text per category.

    Distribution across categories is roughly uniform; topics within a
    category are sampled from that category's hand-curated topic list.
    """
    assert_content_complete(CATEGORIES)
    items: list[Item] = []
    now = datetime.now(timezone.utc).replace(microsecond=0)
    creators = [f"creator_{i:03d}" for i in range(n_creators)]

    for i in range(n):
        category = rng.choice(CATEGORIES)
        spec = CATEGORY_CONTENT[category]
        topic = rng.choice(spec["topics"])
        modifier = rng.choice(MODIFIERS)
        entity = rng.choice(spec["entities"])
        title = rng.choice(spec["title_templates"]).format(topic=topic, modifier=modifier, entity=entity)
        body = rng.choice(spec["body_templates"]).format(topic=topic, modifier=modifier, entity=entity)

        # created_at: last 90 days, biased slightly toward recent (so the
        # "freshness" feature has signal).
        days_ago = int(rng.triangular(0, 90, 20))
        created_at = (now - timedelta(days=days_ago, hours=rng.randint(0, 23))).isoformat()

        items.append(
            Item(
                item_id=f"item_{i:05d}",
                category=category,
                topic=topic,
                title=title,
                body=body,
                creator_id=rng.choice(creators),
                created_at=created_at,
                popularity_score=round(_power_law_popularity(rng), 4),
            )
        )
    return items


def _items_by_category(items: list[Item]) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {c: [] for c in CATEGORIES}
    for it in items:
        out[it.category].append(it)
    return out


def _sample_item(
    pool: list[Item], rng: random.Random, *, popularity_temperature: float = 1.5
) -> Item:
    """Sample one item from a pool, biased by popularity_score.

    `popularity_temperature` > 1 sharpens toward popular items.
    """
    if not pool:
        raise ValueError("empty pool")
    weights = np.array([max(1e-3, it.popularity_score) ** popularity_temperature for it in pool])
    weights /= weights.sum()
    idx = rng.choices(range(len(pool)), weights=weights.tolist(), k=1)[0]
    return pool[idx]


def _sample_event_type(rng: random.Random) -> tuple[str, float]:
    types = list(EVENT_TYPE_DISTRIBUTION.keys())
    weights = list(EVENT_TYPE_DISTRIBUTION.values())
    et = rng.choices(types, weights=weights, k=1)[0]
    return et, EVENT_WEIGHTS[et]


def _random_timestamp_within(days: int, rng: random.Random) -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    delta_seconds = rng.randint(0, days * 24 * 3600)
    return (now - timedelta(seconds=delta_seconds)).isoformat()


def generate_interactions(
    users: list[User],
    items: list[Item],
    rng: random.Random,
    *,
    history_days: int = 30,
) -> list[Interaction]:
    """For each user, sample interactions over the last `history_days`.

    Mix per the design spec:
        70% preferred-category items, 20% related, 10% random.
    Number of interactions per user follows Poisson(activity-level mean).
    """
    by_cat = _items_by_category(items)
    all_items = items
    interactions: list[Interaction] = []

    for u in users:
        mean = ACTIVITY_TO_INTERACTION_MEAN[u.activity_level]
        n_int = max(1, int(np.random.poisson(mean)))

        preferred = [it for c in u.interests for it in by_cat.get(c, [])]
        related_cats = {r for c in u.interests for r in RELATED.get(c, ())}
        related_cats -= set(u.interests)
        related = [it for c in related_cats for it in by_cat.get(c, [])]

        # Defensive fallbacks if a particular pool happens to be empty.
        if not preferred:
            preferred = all_items
        if not related:
            related = all_items

        for _ in range(n_int):
            r = rng.random()
            if r < SOURCE_MIX["preferred"]:
                pool = preferred
            elif r < SOURCE_MIX["preferred"] + SOURCE_MIX["related"]:
                pool = related
            else:
                pool = all_items
            item = _sample_item(pool, rng)
            event_type, weight = _sample_event_type(rng)
            interactions.append(
                Interaction(
                    user_id=u.user_id,
                    item_id=item.item_id,
                    event_type=event_type,
                    event_weight=weight,
                    timestamp=_random_timestamp_within(history_days, rng),
                )
            )
    return interactions


def split_interactions(
    interactions: list[Interaction], *, eval_days: int = 7
) -> tuple[list[Interaction], list[Interaction]]:
    """Split into train (older) and eval (last `eval_days`)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=eval_days)
    train, evalset = [], []
    for it in interactions:
        ts = datetime.fromisoformat(it.timestamp)
        if ts >= cutoff:
            evalset.append(it)
        else:
            train.append(it)
    return train, evalset


# ─── i/o ─────────────────────────────────────────────────────────────────


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")
            n += 1
    return n


def write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


# ─── sanity ──────────────────────────────────────────────────────────────


def sanity_report(
    users: list[User], items: list[Item], train: list[Interaction], evalset: list[Interaction]
) -> dict[str, Any]:
    """Quick stats so you can eyeball the dataset before training."""
    cat_items = Counter(it.category for it in items)
    cat_users = Counter(c for u in users for c in u.interests)
    activity = Counter(u.activity_level for u in users)
    event_types = Counter(i.event_type for i in train)

    # How often does a user's interaction land in one of their preferred categories?
    user_interests = {u.user_id: set(u.interests) for u in users}
    item_cat = {it.item_id: it.category for it in items}
    in_pref = sum(
        1 for i in train if item_cat[i.item_id] in user_interests[i.user_id]
    )
    pref_rate = in_pref / max(1, len(train))

    return {
        "users": len(users),
        "items": len(items),
        "train_interactions": len(train),
        "eval_interactions": len(evalset),
        "items_per_category": dict(cat_items.most_common()),
        "user_interest_distribution": dict(cat_users.most_common()),
        "activity_distribution": dict(activity),
        "event_type_distribution": dict(event_types),
        "preferred_category_hit_rate": round(pref_rate, 4),
    }


# ─── cli ─────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the RecoForge synthetic dataset.")
    p.add_argument("--users", type=int, default=1000)
    p.add_argument("--items", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--history-days", type=int, default=30)
    p.add_argument("--eval-days", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    log.info(
        "generating dataset: users=%d items=%d seed=%d history=%dd eval=%dd",
        args.users, args.items, args.seed, args.history_days, args.eval_days,
    )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    log.info("[1/4] users")
    users = generate_users(args.users, rng)

    log.info("[2/4] items")
    items = generate_items(args.items, rng)

    log.info("[3/4] interactions")
    interactions = generate_interactions(users, items, rng, history_days=args.history_days)
    train, evalset = split_interactions(interactions, eval_days=args.eval_days)

    log.info("[4/4] writing files → %s", args.out)
    n_users = write_jsonl(args.out / "users.jsonl", users)
    n_items = write_jsonl(args.out / "items.jsonl", items)
    n_train = write_jsonl(args.out / "interactions.jsonl", train)
    n_eval = write_jsonl(args.out / "interactions_eval.jsonl", evalset)

    report = sanity_report(users, items, train, evalset)
    meta = {
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "users.jsonl": n_users,
            "items.jsonl": n_items,
            "interactions.jsonl": n_train,
            "interactions_eval.jsonl": n_eval,
        },
        "config": {
            "history_days": args.history_days,
            "eval_days": args.eval_days,
            "source_mix": SOURCE_MIX,
            "event_type_distribution": EVENT_TYPE_DISTRIBUTION,
        },
        "report": report,
    }
    write_meta(args.out / "dataset_meta.json", meta)

    log.info("done.")
    log.info("  users:        %d", n_users)
    log.info("  items:        %d", n_items)
    log.info("  train interactions: %d", n_train)
    log.info("  eval interactions:  %d", n_eval)
    log.info("  preferred-category hit rate: %.1f%%", report["preferred_category_hit_rate"] * 100)
    log.info("  → meta written to %s", args.out / "dataset_meta.json")


if __name__ == "__main__":
    main()
