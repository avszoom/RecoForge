# RecoForge

A real-time recommendation system POC. An offline two-tower model learns long-term user/item embeddings; the online layer adapts recommendations to the user's current session within milliseconds; brand-new users and items are recommendable immediately via cold-start fallbacks.

> **Source of truth for the design:** this README + [`src/models/README.md`](src/models/README.md) + [`data/README.md`](data/README.md).

---

## Architecture

```
                        Synthetic Dataset
                  ┌──────────────────────────┐
                  │  users.jsonl             │
                  │  items.jsonl             │
                  │  interactions.jsonl      │
                  └────────────┬─────────────┘
                               │
                               ▼
                       Offline Training
                  ┌──────────────────────────┐
                  │  two-tower model (PyTorch)│
                  │  user_emb / item_emb 64-d│
                  │  frozen MiniLM for text  │
                  │  in-batch sampled softmax│
                  └────────────┬─────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │  FAISS item index        │
                  │  user_embeddings.npy     │
                  │  item_embeddings.npy     │
                  └────────────┬─────────────┘
                               │
                               ▼
                       Online Serving (Streamlit)
                  ┌──────────────────────────┐
                  │  long-term + session blend│
                  │  multi-source candidates │
                  │  rank + re-rank          │
                  │  add new user / new item │
                  │  click → live update     │
                  └──────────────────────────┘
```

**Two-tower idea in one sentence:** train a user encoder and an item encoder so that `dot(user_emb, item_emb)` predicts engagement. Items are precomputed once → FAISS finds nearest items per user in milliseconds. The user side blends a frozen long-term embedding with a real-time session embedding, so recommendations adapt as the user clicks without retraining.

---

## Quickstart

```bash
# 1. install deps (Python 3.10+)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. generate the synthetic dataset (1k users, 4k items, ~84k interactions)
python -m src.data.generate_dataset

# 3. run the test suite (per-file isolation; see scripts/test.sh for why)
./scripts/test.sh

# 4. train the two-tower model
python -m src.models.text_features         # ~5s on MPS, frozen MiniLM cache
python -m src.models.train_two_tower       # ~25s on MPS, 10 epochs
python -m src.models.export_embeddings     # exports user/item embeddings
python -m src.indexing.build_faiss         # builds artifacts/item_index.faiss

# 5. produce a recommendation
python -m src.serving.recommender u_0042                 # adaptive (default)
python -m src.serving.recommender u_0042 --mode long_term

# 6. record a click and watch the recs shift
python -m src.serving.recommender u_0042 --click item_01234

# 7. cold-start a brand-new user / item (Phase 6)
python -m src.serving.add_user --interests "AI Infrastructure" \
    --age-bucket 25-34 --location US --activity-level high --show-recs
python -m src.serving.add_item --category Travel \
    --title "A weekend in Porto" --body "Cheap flights, walkable streets..."

# 8. launch the Streamlit demo (use the wrapper — see Notes below)
./scripts/run_app.sh
# Then open http://localhost:8501 — five pages in the sidebar:
#   1. Recommendations          — pick a user, click items, watch recs adapt
#   2. Item explorer            — browse the catalog by category, click to feed session
#   3. Add new item             — Phase 6 cold-start with verification
#   4. User state debugger      — long_term vs adaptive side-by-side
#   5. Evaluation               — Phase 8 placeholder

# NOTE for macOS Apple Silicon users: launch via ./scripts/run_app.sh, NOT
# `streamlit run app/streamlit_app.py` directly. The wrapper exports
# OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE BEFORE python starts, which
# sidesteps a pthread_mutex_init segfault that otherwise hits when the
# faiss-cpu and torch libomp instances coexist in the same process.
```

---

## Repo layout

```
recommendationForge/
├── README.md                       ← you are here (project overview)
├── requirements.txt
├── .gitignore
│
├── data/                           ← generated JSONL (gitignored)
│   ├── README.md                   ← dataset spec + schemas
│   ├── users.jsonl
│   ├── items.jsonl
│   ├── interactions.jsonl
│   ├── interactions_eval.jsonl     ← held-out last 7 days
│   └── dataset_meta.json
│
├── src/
│   ├── data/
│   │   ├── constants.py            ← taxonomy, weights, source mix
│   │   ├── content_templates.py    ← per-category title/body fragments
│   │   └── generate_dataset.py     ← CLI to (re)build the dataset
│   │
│   ├── models/
│   │   ├── README.md               ← two-tower architecture + training
│   │   ├── two_tower.py            ← (Phase 2) PyTorch model definition
│   │   ├── train_two_tower.py      ← (Phase 2) training loop
│   │   └── export_embeddings.py    ← (Phase 2) batch inference
│   │
│   ├── indexing/
│   │   └── build_faiss.py          ← (Phase 3) FAISS IndexFlatIP
│   │
│   ├── serving/
│   │   ├── recommender.py          ← (Phase 4-5) request-time pipeline
│   │   ├── user_state.py           ← (Phase 5) session embeddings
│   │   ├── candidate_generators.py ← (Phase 5) ANN / trending / fresh / category
│   │   ├── ranker.py               ← (Phase 5) scoring + re-ranking rules
│   │   ├── add_item.py             ← (Phase 6) cold-start item insertion
│   │   └── add_user.py             ← (Phase 6) cold-start user insertion
│   │
│   └── evaluation/
│       ├── metrics.py              ← (Phase 8) Recall@k, MRR, NDCG@k
│       └── evaluate.py             ← (Phase 8) baseline comparison
│
├── app/
│   └── streamlit_app.py            ← (Phase 7) 4-page UI
│
├── artifacts/                      ← (gitignored) trained model + embeddings
│   ├── two_tower.pt
│   ├── user_embeddings.npy
│   ├── item_embeddings.npy
│   ├── item_index.faiss
│   └── item_mapping.json
│
└── tests/
    └── test_dataset.py             ← schema + integrity (4 tests)
```

---

## Status

- [x] **Phase 0** — Scaffold + requirements
- [x] **Phase 1** — Synthetic dataset generator (`src/data/`)
- [x] **Phase 2** — Two-tower model + training (`src/models/`) — see [Known limitations](src/models/README.md#known-limitations-v1)
- [x] **Phase 3** — FAISS item index (`src/indexing/`)
- [x] **Phase 4** — Long-term-only recommender (`src/serving/recommender.py`)
- [x] **Phase 5** — Online session adaptation + candidate generators
- [x] **Phase 6** — Add new user / new item flows (cold start)
- [x] **Phase 7** — Streamlit UI (4 pages)
- [x] **Phase 8** — Evaluation suite + baseline comparison

### Phase 8 results (full 19,681-interaction eval split)

|                          | recall@10 | mrr@10 | ndcg@10 |
|---|---:|---:|---:|
| popularity               | 0.0219    | 0.0068 | 0.0102  |
| category (declared interests + popularity) | 0.1283 | 0.0369 | 0.0578  |
| two_tower (long_term)    | 0.0118    | 0.0036 | 0.0054  |
| **two_tower + session-replay** | **0.0235** | 0.0061 | 0.0100  |

The headline finding: **session blending ~doubles recall@10 over long-term-only** (0.0118 → 0.0235), validating the Phase 5 design. The category baseline beats the two-tower on raw recall — its candidate pool is much smaller (~300-600 items vs the full 4000) so high popularity items dominate. Within-category discrimination is the open weakness (see [`src/models/README.md` known-limitations](src/models/README.md#known-limitations-v1)). Absolute numbers will rise to 8-15% with real (non-templated) text + V2 negative sampling.

---

## Why this design?

| Decision | Why |
|---|---|
| **Two-tower + FAISS** | Item side is precomputed; only the user side runs at request time. Industry standard for sub-100ms recommenders. |
| **Frozen MiniLM for item text** | Pretrained on billions of sentence pairs — already places "AI Infrastructure" near "Programming" in semantic space. We don't have the data to fine-tune it; we don't need to. |
| **64-d trained embeddings** | Plenty of capacity for 4k items, FAISS searches stay instant, training fits in CPU minutes. |
| **In-batch sampled softmax** | Standard contrastive objective. Other items in the batch act as free negatives — no separate negative-sampler needed. |
| **`<UNK>` row in lookup tables** | Lets the trained towers handle never-seen-before users/items by falling back to the rest of their features. The bridge to a proper retrain. |
| **0.7 long-term + 0.3 session blend** | Anchored personalization that still moves on every click. Dynamic weights for new users (more session weight when long-term is unreliable). |
| **Streamlit single-process** | Same as the llmrouter POC. Free deploy, FAISS-cpu works there, no Redis. |

---

## See also

- [`data/README.md`](data/README.md) — what the synthetic dataset contains and how it is generated.
- [`src/models/README.md`](src/models/README.md) — two-tower architecture, training objective, known limitations + V2 roadmap.
- [`src/indexing/README.md`](src/indexing/README.md) — FAISS index design + `ItemIndex` API.
- [`src/serving/README.md`](src/serving/README.md) — Phase 4 (long-term recommender), Phase 5 (online adaptation, candidate generators, ranker, diversifier), Phase 6 (cold start), CLI cheat sheet, test infrastructure notes.
