# Serving layer — Phases 4, 5, 6

Everything that happens at request time. Reads the trained artifacts from Phase 2 + the FAISS index from Phase 3 and turns a `user_id` (or fresh signup details) into a ranked list of items.

> **Status:** Phase 4, 5, 6 all done. Phase 7 (Streamlit UI) sits on top of this layer.

```
   user_id ─┐
            ▼                 ┌── trained user_embedding (Phase 2)
        Recommender           │
            │                 ├── trained item embeddings  (Phase 2)
            │       loads     │
            ├─────────────────┼── FAISS item index         (Phase 3)
            │                 │
            │                 ├── item / user catalogs     (data/*.jsonl)
            │                 │
            │                 ├── interaction history      (filter & trending)
            │                 │
            │                 └── UserStateStore           (recent clicks)
            │
            ▼
       recommend(user_id, k, mode)
                          │
                          ├── mode='long_term'  → Phase 4 (FAISS only)
                          └── mode='adaptive'   → Phase 5 (5 generators
                                                  + ranker + diversify)
       on_click(user_id, item_id)
                          │
                          ▼
            UserState.record_click + trending++
            → next recommend reflects the click
       add_user(...)  / add_item(...)        Phase 6 cold start
                          │
                          ▼
            cold_start tower forward with id=<UNK>
            → entity is recommendable in milliseconds
```

---

## File layout

```
src/serving/
├── recommender.py            Recommender class + CLI (this is the entrypoint)
├── candidate_generators.py   five generator functions for Phase 5
├── ranker.py                 merge → linear score → 80/10/10 diversify
├── user_state.py             UserState + UserStateStore (online clicks)
├── cold_start.py             ColdStartEngine (lazy torch + MiniLM)
├── add_user.py               CLI: cold-start a new user
└── add_item.py               CLI: cold-start a new item
```

---

# Phase 4 — long-term-only recommender

The simplest end-to-end pipeline. Single line:

```
user_id  →  load user_embedding  →  FAISS top-k  →  filter seen  →  k items
```

**API:**
```python
from src.serving.recommender import Recommender
rec = Recommender()
rec.recommend("u_0042", k=10, mode="long_term")
```

**CLI:**
```bash
python -m src.serving.recommender u_0042 --mode long_term --k 10
```

**What it does:**
- Loads `artifacts/user_embeddings.npy[user_id]` (the trained long-term vector)
- `index.search(user_emb, k=k*3+50)` (over-fetch for filtering)
- Drops items in the user's `seen_items()` set (built from `interactions.jsonl` + `interactions_eval.jsonl` + `online_interactions.jsonl`)
- Returns the first `k` survivors as `Recommendation` records

**What it does NOT do:**
- No session blending (recent clicks ignored)
- No multi-source candidates (only ANN)
- No ranking layer (FAISS scores ARE the ranking)
- No re-ranking rules

This is the floor — useful as a baseline against the adaptive mode in Phase 5.

---

# Phase 5 — online adaptation (the heart of the system)

What makes "real-time" real. Same `recommend()` call, `mode="adaptive"` (the default).

```
              ┌───────────────────────────────────────────────────┐
              │  long-term embedding  (Phase 2, frozen)           │
   blend ←────┤                                                   │
              │  session embedding   (mean of recent clicks)      │
              └───────────────────────────────────────────────────┘
                                  │
                          final_user_embedding
                                  │
                ┌─────────────────┼──────────────────┐
                ▼                 ▼                  ▼
          ┌───────────┐    ┌───────────┐    ┌──────────────┐
          │ ann_long  │    │ similar_  │    │ trending /   │
          │ _term     │    │ to_recent │    │ fresh /      │
          │ (FAISS)   │    │ (FAISS×N) │    │ category     │
          └─────┬─────┘    └─────┬─────┘    └──────┬───────┘
                └─────────────┬──┴─────────────────┘
                              ▼
                            merge  (dedupe, keep all source scores)
                              │
                              ▼
                            rank   (linear scorer)
                              │
                              ▼
                          diversify (8 personalized + 1 trending + 1 fresh)
                              │
                              ▼
                          list[Recommendation]
```

## The five candidate generators

Each generator returns `Candidate(item_id, source, source_score in [0,1])`.

| Generator | What it returns | Score |
|---|---|---|
| `ann_long_term` | Top-50 from FAISS using `final_user_embedding`. Workhorse — most candidates land here. | cosine, mapped to [0, 1] |
| `similar_to_recent` | For the user's last 5 clicked items, FAISS top-10 each. Aggregated by max similarity per item. | cosine, mapped to [0, 1] |
| `trending` | Top items by event-weighted click count (counter seeded from offline log + bumped by `on_click`). | count / max_count |
| `fresh` | Items created in the last 7 days, sorted by age. | linear decay 1.0 → 0.0 over 7d |
| `category_interest` | Items from the user's declared interests, ranked by `popularity_score`. | popularity / max |

## Dynamic user-embedding blend

The user embedding fed into `ann_long_term` is not the static long-term vector — it's a blend that responds to recent clicks:

```python
session_emb = mean(item_embeddings[recent_clicks])    # rebuilt per request
session_emb = session_emb / ||session_emb||

# auto_blend_weights(n_session, n_history) — cold-start aware
if n_history < 5:                                # cold-start user (signed up via add_user)
    n_session < 3   → (0.3, 0.7)                 # rely on first clicks heavily
    n_session < 10  → (0.5, 0.5)
    else            → (0.7, 0.3)
else:                                            # established user (in trained model)
    always          → (0.7, 0.3)                 # anchor on long-term

final_user_emb = (w_long * long_term + w_session * session_emb).normalize()
```

**Why gate the aggressive schedule on `n_history`:** for established users — the common case in the synthetic dataset, where everyone has 50–200 logged interactions — a single off-pattern click shouldn't redirect their whole feed. The session-heavy schedule is reserved for cold-start users whose long-term embedding is a synthesized `<UNK>` fallback.

**Manual override:** `recommend(..., blend=(0.6, 0.4))` overrides the schedule. The Streamlit Recommendations page exposes this as a "Long-term weight" slider in the sidebar (toggle off "Auto blend" to manually drag the weight).

## Linear ranker

Defined in `ranker.py`. Per the design spec:

```
final_score =
    0.45 * ann_score
  + 0.25 * recent_similarity_score
  + 0.15 * category_match (1.0 if item.category in user.interests else 0)
  + 0.10 * trending_score
  + 0.05 * freshness_score
```

Items missing a particular source contribute 0 for that term. So an item only present in the `trending` generator scores `0.10 * trending_score` — they only beat ANN candidates when ANN-only items have very low similarity.

## 80/10/10 diversifier

After ranking, `diversify()` composes a 10-slot page as **8 personalized + 1 trending + 1 fresh**:

1. **Slots 1–8**: take the top of the ranked list (any source, but in practice ann + recent dominate).
2. **Slot 9**: best `trending` candidate not already in the page.
3. **Slot 10**: best `fresh` candidate not already in the page.
4. If trending or fresh is empty, backfill from the ranked list.

Seen items are filtered at every step. The composition keeps the page diverse while still being driven by the linear ranker.

## On-click flow

```python
rec.on_click("u_0042", "item_01234")
```

Updates:
- `UserState.recent_clicked_items` (capped at 20, oldest evicted)
- `UserState.recent_categories` Counter
- `UserState.session_embedding` invalidated (recomputed on next recommend)
- `trending_counter[item_id] += 1`
- `user_history[user_id].add(item_id)` (so it won't recommend it again)
- Appends to `data/online_interactions.jsonl`
- Atomically saves `artifacts/user_state.json`

No model retraining. The session shift is purely vector arithmetic — recommendation latency is unchanged.

## Demo (real run)

```
u_0007  declared interests = ['Programming', 'Startups']  (NO Travel)

BEFORE clicks:
   1  Programming  ann       'Practical profiling tools: for beginners'
   2  Programming  ann       'git rebase workflows for working engineers'
   ...
   9  Gaming       trending  'Why remasters keep working'
  10  Startups     fresh     'pricing strategy mistakes...'
   → 9/10 in declared interests

AFTER clicking 3 Travel items:
   1  Travel       ann,recent      'rail passes in Edinburgh'
   2  Travel       ann,recent      'A guide to shoulder season trips...'
   3  Travel       ann,recent      'budget Europe in Tokyo'
   ...
   9  Gaming       trending        (still injected)
  10  Startups     fresh           (still injected)
   → 7/10 are Travel
```

The shift happened in **one recommend call** after the third click — no retraining.

---

# Phase 6 — cold start (new user / new item)

Brand-new entities aren't in the trained id-lookup tables. We use the `<UNK>` row mechanism (Phase 2) to produce a usable embedding from the rest of the features.

## How `<UNK>` makes this work

During training (Phase 2), 1% of rows had their `user_id` or `item_id` replaced with row 0 (`<UNK>`). The MLP learned to lean on the rest of the features (text, category, popularity, freshness for items; activity, declared interests for users) when the id signal is missing.

At cold-start time, we just pass `id=0` along with the entity's real features:

```python
# Item tower with item_id=<UNK>:
new_item_emb = model.item_tower(
    item_id=tensor([0]),                  # <UNK>
    category=tensor([cat_to_idx[category]]),
    text_emb=minilm.encode(title + body),  # frozen MiniLM, real signal
    popularity=tensor([0.0]),
    freshness=tensor([1.0]),
)

# User tower with user_id=<UNK>:
new_user_emb = model.user_tower(
    user_id=tensor([0]),                  # <UNK>
    activity=tensor([act_to_idx[activity_level]]),
    preferred_multihot=interest_multihot, # the user's declared interests
)
```

The output is a 64-d vector that lives in the same space as known users/items, so FAISS treats it identically.

## `add_user`

```python
new_uid = rec.add_user(
    interests=["AI Infrastructure"],
    age_bucket="25-34",
    location="US",
    activity_level="high",
)
# user is now recommendable
recs = rec.recommend(new_uid, k=10, mode="adaptive")
```

CLI:
```bash
python -m src.serving.add_user --interests "AI Infrastructure" \
    --age-bucket 25-34 --location US --activity-level high --show-recs
```

What gets persisted (when `persist=True`, the default):
- Append to `data/users.jsonl`
- `np.save("artifacts/user_embeddings.npy", user_emb)` — the array grew by 1
- `json.dump(user_id_to_row, "artifacts/user_id_to_row.json")` — added the new id

What's NOT touched:
- The trained model (`two_tower.pt`) — unchanged
- The interactions log — empty until the user starts clicking

## `add_item`

```python
new_iid = rec.add_item(
    category="Travel",
    title="A weekend in Porto: cheap flights, river views",
    body="Three days, walkable streets, food that ruins you for home.",
)
# new item is now retrievable from FAISS
```

CLI:
```bash
python -m src.serving.add_item --category Travel \
    --title "A weekend in Porto" --body "..." --recs-for u_0007
```

What gets persisted:
- Append to `data/items.jsonl`
- FAISS `index.add(emb)` + persist the updated `item_index.faiss` and `item_mapping.json`
- `np.save("artifacts/item_embeddings.npy", item_emb)` — grew by 1
- `json.dump(item_id_to_row, "artifacts/item_id_to_row.json")` — added the new id

In-memory updates (no separate persistence):
- `items_by_id[new_iid] = record`
- `items_by_category[category]` gets the new item appended
- `fresh_items_sorted` gets the new item at age 0 (at the head)

The new item appears in `ann_long_term`, `similar_to_recent`, `fresh`, and `category_interest` candidates immediately.

## Demo

```python
new_uid = rec.add_user(interests=["AI Infrastructure"], ...)
recs = rec.recommend(new_uid, k=10, mode="adaptive")
# → 10/10 AI Infrastructure (cold-start user, never trained on)

new_iid = rec.add_item(category="Travel", title="A weekend in Porto", ...)
recs = rec.recommend("u_0003", k=20, mode="adaptive")  # u_0003 likes Travel
# → new item appears at rank 2, sources=['ann', 'fresh'], score=0.617
```

## Cold-start engine notes

- **Lazy load**: `ColdStartEngine` doesn't load `two_tower.pt` or MiniLM at process start. Only the first `add_user` / `add_item` call pays the ~3-second cost. Phase 4 / 5 paths never touch torch.
- **Library order**: `cold_start.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time. If you're embedding the Recommender in another process that already loaded faiss, ensure torch is imported BEFORE faiss to avoid a `load_state_dict` segfault on macOS Apple Silicon.

---

# Recommendation record

Every recommend call returns `list[Recommendation]`:

```python
@dataclass
class Recommendation:
    item_id: str
    score: float                # final ranker score (or cosine in long_term mode)
    title: str
    category: str
    topic: str
    body: str
    popularity: float
    created_at: str             # ISO-8601
    rank: int                   # 1-indexed
    sources: list[str]          # ["ann", "trending", "fresh"]
    source_scores: dict[str, float]
```

`sources` enables source-badge rendering in the UI without extra lookups.

---

# CLI cheat sheet

```bash
# Show top-10 adaptive recs (default mode)
python -m src.serving.recommender u_0042

# Phase-4 baseline (no session blend, no diversify)
python -m src.serving.recommender u_0042 --mode long_term

# Show profile + recs for a random user
python -m src.serving.recommender --random --seed 7

# Record a click and immediately reshow recs
python -m src.serving.recommender u_0042 --click item_01234

# Reset a user's session state (recent clicks → empty)
python -m src.serving.recommender u_0042 --reset

# Cold-start a new user (Phase 6) and immediately recommend for them
python -m src.serving.add_user --interests "Programming" "AI Infrastructure" \
    --age-bucket 25-34 --location US --activity-level high --show-recs

# Cold-start a new item and verify it shows up for a relevant user
python -m src.serving.add_item --category Travel \
    --title "A weekend in Porto" --body "Cheap flights, walkable streets..." \
    --recs-for u_0007
```

---

# Tests

| File | Tests | What it covers |
|---|---:|---|
| `tests/test_recommender.py` | 5 | Phase 4: load shapes, k-items sorted, filter_seen contract (basic + injected), unknown user error, single-interest match rate. |
| `tests/test_phase5.py` | 10 | Phase 5: UserState capping + invalidation, `UserStateStore` round-trip, `final_user_embedding` blend, on_click side effects, adaptive recs shift after clicks (≥5/10 after 5 foreign-category clicks), source tags populated, long_term mode unchanged. |
| `tests/test_phase6.py` | 10 | Phase 6: array growth on add, FAISS index extension, fresh-pool insertion, validation of inputs, cold-start user recall ≥80%, cold-start item retrievable, end-to-end persistence to disk. |

Run via the per-file isolation script (see Notes below):

```bash
./scripts/test.sh
```

---

# Notes on the macOS faiss + torch combo

faiss-cpu and PyTorch each ship their own libomp. On macOS Apple Silicon, having both loaded into a long-lived process can intermittently segfault `faiss.search` after PyTorch has loaded a state dict. Mitigations applied:

- `KMP_DUPLICATE_LIB_OK=TRUE` set in `cold_start.py` and `recommender.py` at import time.
- `tests/test_phase6.py` imports `torch` BEFORE the Recommender (forces torch's libomp to load first).
- `scripts/test.sh` runs each test file in its own pytest invocation (separate processes), so test_indexing.py never has torch loaded into its process and test_phase6.py is the only file where both libraries coexist.

When deploying to Streamlit Cloud (Phase 7), this isn't a concern — Streamlit's process loads everything once at startup, and the import-order fix in `tests/test_phase6.py` is mirrored in `app/streamlit_app.py`.
