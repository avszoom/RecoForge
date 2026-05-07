# Dataset

Synthetic data for the RecoForge POC. Three JSONL files plus a held-out evaluation split, all regenerable from a single seed.

> **Why synthetic?** No production logs. We need a dataset where the ground-truth structure (which categories a user prefers, which items they should engage with) is known by construction, so we can verify that the two-tower model learns *something real* rather than overfitting to noise.

---

## What's in here

| File | Rows (default) | Purpose |
|---|---:|---|
| `users.jsonl` | 1,000 | User profiles with declared interests + activity level |
| `items.jsonl` | 4,000 | Items with templated title/body, category, popularity, freshness |
| `interactions.jsonl` | ~64,000 | Training interactions over the last ~30 days (excluding the eval window) |
| `interactions_eval.jsonl` | ~20,000 | Held-out interactions from the **last 7 days** |
| `dataset_meta.json` | 1 | Seed, counts, sanity stats from the generator run |

All files are gitignored. To regenerate:

```bash
python -m src.data.generate_dataset                     # defaults
python -m src.data.generate_dataset --seed 7            # different draw
python -m src.data.generate_dataset --users 500 --items 2000   # smaller
```

---

## Schemas

### `users.jsonl`

```json
{
  "user_id": "u_0042",
  "age_bucket": "25-34",
  "location": "EU",
  "interests": ["AI Infrastructure", "Programming"],
  "activity_level": "high"
}
```

| Field | Type | Notes |
|---|---|---|
| `user_id` | string | Stable, zero-padded (`u_0000` … `u_0999`). |
| `age_bucket` | string | One of `18-24` / `25-34` / `35-44` / `45-54` / `55+`. |
| `location` | string | One of `US` / `EU` / `UK` / `IN` / `APAC` / `LATAM`. |
| `interests` | list[string] | 1–3 categories, sampled from the canonical taxonomy. |
| `activity_level` | string | `low` / `medium` / `high` — drives interaction count. |

### `items.jsonl`

```json
{
  "item_id": "item_00123",
  "category": "AI Infrastructure",
  "topic": "LLM inference cost",
  "title": "Reducing LLM inference cost: from first principles",
  "body": "LLM inference cost has become a critical bottleneck. Teams using vLLM are reporting 3x improvements...",
  "creator_id": "creator_087",
  "created_at": "2026-04-21T14:30:55+00:00",
  "popularity_score": 0.4612
}
```

| Field | Type | Notes |
|---|---|---|
| `item_id` | string | Zero-padded (`item_00000` … `item_03999`). |
| `category` | string | One of the 12 canonical categories. |
| `topic` | string | A category-specific topic from the templated content pool. |
| `title` / `body` | string | Templated text, deterministic per seed. Will be embedded by frozen MiniLM at training time. |
| `creator_id` | string | One of 200 synthetic creators. |
| `created_at` | ISO-8601 | Distributed across the last 90 days, biased slightly toward recent (so the freshness feature has signal). |
| `popularity_score` | float [0, 1] | Power-law distributed: a few items hit 0.8+, most are middling, long tail near 0. |

### `interactions.jsonl` / `interactions_eval.jsonl`

```json
{
  "user_id": "u_0042",
  "item_id": "item_00123",
  "event_type": "click",
  "event_weight": 1.0,
  "timestamp": "2026-04-21T16:11:02+00:00"
}
```

| Field | Type | Notes |
|---|---|---|
| `user_id` / `item_id` | string | Foreign keys into the other two files. Referential integrity is enforced by tests. |
| `event_type` | string | `view` / `click` / `like` / `save` / `share`. |
| `event_weight` | float | Always derived from `event_type`: 0.2 / 1.0 / 2.0 / 3.0 / 4.0 respectively. The training loop weights positive pairs by this. |
| `timestamp` | ISO-8601 | Distributed over the last 30 days. Anything from the **last 7 days** is in `interactions_eval.jsonl`; everything older is in `interactions.jsonl`. |

---

## Generation logic

### 1. Taxonomy (`src/data/constants.py`)

12 categories, each with 2–3 hand-curated "related" categories. Used by:

- **users.jsonl**: each user's `interests` is sampled from this list.
- **items.jsonl**: each item is assigned to one category.
- **interactions.jsonl**: the source-mix logic below uses the related-category adjacency.

### 2. Items: text + popularity

`src/data/content_templates.py` holds 8 topics × 4 title templates × 3 body templates **per category**. Combined with rotating modifiers ("from first principles", "in 2026", …) and entities ("Vercel", "Stripe", …), this yields plenty of distinct titles without repeats — and crucially, items in the same category share vocabulary, so a frozen MiniLM encoder will place them near each other in semantic space.

Popularity is a **power-law draw** (Pareto, capped to [0, 1]). About 5% of items end up with `popularity_score > 0.5`; most are well below 0.2. This shapes which items the interaction generator picks: when sampling within a category, the weight is `popularity ** 1.5`, so popular items get exponentially more clicks.

### 3. Interactions: the source-mix rule

For each user, draw `N` interactions where `N ~ Poisson(activity_mean)`:

| activity_level | mean | typical N |
|---|---:|---|
| `low` | 30 | ~20–40 |
| `medium` | 80 | ~65–95 |
| `high` | 200 | ~170–230 |

For each interaction, pick the source pool:

```
70% → items from the user's preferred categories
20% → items from related categories (per the adjacency map)
10% → any item, uniform — exploration noise
```

Within the chosen pool, sample one item weighted by popularity. Pick an event type from the empirical distribution (mostly views/clicks, rare shares), and stamp a random timestamp within the last 30 days.

The 70/20/10 mix is *the* signal the two-tower model learns. The post-hoc sanity check in `dataset_meta.json` reports the actual hit rate — should land at ~70%, with ≥ 60% as the test threshold.

### 4. Train/eval split

Last 7 days of interactions → `interactions_eval.jsonl`. Everything older → `interactions.jsonl`. **This is a temporal split, not random** — at evaluation time we want to see whether the model can predict the *future* engagement of a user given their *past* engagement, which is closer to the real production task.

---

## Sanity stats from a default run (seed=42)

```
users:        1,000
items:        4,000
train interactions: 64,437
eval  interactions: 19,681
preferred-category hit rate: 71.7%        ← target: 70%, threshold: ≥ 60%

items per category:
  AI Infrastructure: 367   Programming: 378   Movies & TV: 336
  Music: 331   Food: 330   Health & Fitness: 324  ...

activity distribution:
  low: 397    medium: 403    high: 200    ← target: 40 / 40 / 20

event-type distribution (training):
  view: 29,016  click: 22,492  like: 7,764  save: 3,225  share: 1,940
```

---

## Tests

`tests/test_dataset.py` enforces:

- Every record has the required schema fields.
- All categories, locations, age buckets, activity levels, and event types are from the canonical sets.
- No duplicate `user_id` or `item_id`.
- Every interaction's `user_id` and `item_id` exists in the other files (no dangling refs).
- The preferred-category hit rate is ≥ 60%.

```bash
pytest tests/ -q
```

---

## Future work

- Replace synthetic data with real interaction logs (the `<UNK>` cold-start mechanic is designed to absorb new users/items into the same training set).
- Add seasonality and time-of-day patterns to interactions (currently uniform-random in time).
- Add a few "evergreen" items whose popularity doesn't decay, to test whether the freshness feature over-penalizes them.
