# Two-tower model

The trained model at the heart of RecoForge. Two small neural networks — one that turns a user into a 64-d vector, one that turns an item into a 64-d vector — such that **`dot(user_emb, item_emb)` predicts engagement.**

> **Status:** design doc. Phase 2 implementation lands `two_tower.py`, `train_two_tower.py`, and `export_embeddings.py` in this directory. Numbers in this doc are the targets we'll validate against.

---

## The architecture in one picture

```
   user features                        item features
        │                                    │
        ▼                                    ▼
  ┌──────────────┐                    ┌──────────────┐
  │  User tower  │                    │  Item tower  │
  │     (MLP)    │                    │     (MLP)    │
  └──────┬───────┘                    └──────┬───────┘
         │                                   │
         ▼                                   ▼
  user_emb  (64-d, L2-norm)         item_emb (64-d, L2-norm)
         │                                   │
         └──────────── · ────────────────────┘
                       │
                       ▼
                 score (scalar)
            "how much will user like item"
```

**Two towers, not one model.** At serving time we want to compute the user-vector once, then match it against thousands of item-vectors via FAISS. If user and item were entangled in one model, we'd need a forward pass per `(user, item)` pair — millions per request. With separated towers, items are precomputed; only the user side runs at request time.

The score function is just `dot(user_emb, item_emb)`. Both vectors are L2-normalized, so dot product is equivalent to cosine similarity.

---

## What each tower sees

### User tower

| Input | Type | How |
|---|---|---|
| `user_id` | int | `nn.Embedding(num_users + 1, 32)`. Row 0 reserved for `<UNK>`. |
| `activity_level` | int | `nn.Embedding(3, 8)` over `low` / `medium` / `high`. |
| `preferred_categories` | list[int] | Mean of category embeddings (`nn.Embedding(num_categories, 16)`). |

```
  concat(user_id_emb, activity_emb, mean_cat_emb)        → 56-d
  Linear(56 → 128) + ReLU + Dropout(0.1)
  Linear(128 → 64)
  L2-normalize
```

### Item tower

| Input | Type | How |
|---|---|---|
| `item_id` | int | `nn.Embedding(num_items + 1, 32)`. Row 0 reserved for `<UNK>`. |
| `category` | int | `nn.Embedding(num_categories, 16)`. |
| `text` (title + body) | float[384] | **Frozen** MiniLM embedding — precomputed once, fed in as a fixed feature. |
| `popularity` | float | Scalar, normalized to [0, 1]. |
| `freshness` | float | Scalar in [0, 1]: `exp(-age_days / 14)` so freshness decays smoothly. |

```
  concat(item_id_emb, category_emb, text_emb, popularity, freshness)   → 434-d
  Linear(434 → 256) + ReLU + Dropout(0.1)
  Linear(256 → 64)
  L2-normalize
```

**Total trainable params:** ~200k (small lookup tables + two MLPs). Trains end-to-end on CPU in ~5 minutes.

---

## Why frozen MiniLM for item text

`sentence-transformers/all-MiniLM-L6-v2` is a 384-d sentence encoder pretrained on billions of sentence pairs. It already places "LLM inference cost" near "GPU scheduling" in its embedding space.

Two reasons we use it frozen:

1. **We don't have the data to fine-tune it.** 64k interactions over 4k items isn't enough to retrain a 22M-param transformer without overfitting.
2. **It would blow up our compute.** Frozen MiniLM embeddings are computed once, cached as a `(4000, 384)` numpy array, and fed in as a fixed feature. Training only updates the tiny MLPs on top. End-to-end fine-tuning would multiply training cost by ~100x.

The MLP that follows MiniLM is what learns "how to project MiniLM's general-purpose semantic space into the 64-d space where this dataset's user-item geometry makes sense."

---

## Training objective: in-batch sampled softmax

For each batch of 128 positive `(user, item)` pairs from `interactions.jsonl`, weighted by `event_weight`:

```
1.  Run user tower on all 128 users  → U: shape (128, 64)
2.  Run item tower on all 128 items  → I: shape (128, 64)
3.  Compute the score matrix          → S = U @ I.T,  shape (128, 128)
                                                       ▲
                       row i = user_i's score against ALL 128 items
                       in the batch — only S[i, i] is the real positive,
                       the other 127 are sampled "negatives" for free.

4.  Apply softmax row-wise → probability distribution per row, summing to 1.
5.  Cross-entropy loss with target = diagonal:
        loss_i = -log( softmax(S[i, :])[i] )
        loss   = mean(loss_i × event_weight_i)

6.  Adam step, repeat.
```

> See the project README's "softmax" section for the simple-words explanation. The short version: softmax turns the row of scores into probabilities that sum to 1, and the loss is "how much probability mass landed on the right answer." Backprop pushes the network to make the diagonal score much higher than the other 127.

**Why this works:** every batch gives us one positive + 127 negatives essentially for free. No separate negative sampler, no hard-negative mining for v1. The model is forced to learn that the *specific* item this user engaged with should rank above 127 random other items — which is exactly the retrieval task we'll do at serving time.

---

## Cold-start mechanics: the `<UNK>` row

The lookup tables for `user_id` and `item_id` reserve **row 0** as a special "unknown" token. During training, we randomly replace ~1% of `user_id`s and ~1% of `item_id`s with row 0:

```python
# inside the training loop
mask_users = torch.rand(batch_user_ids.shape) < 0.01
mask_items = torch.rand(batch_item_ids.shape) < 0.01
batch_user_ids[mask_users] = 0    # <UNK>
batch_item_ids[mask_items] = 0    # <UNK>
```

This trains the MLPs to handle the case where the ID embedding contains no useful information — they learn to lean on the *other* features (activity + preferred categories for users; text + category + popularity for items).

At serving time, for a never-seen-before user or item, we just pass `id=0`. The tower produces a sensible 64-d vector based on the rest of the features. The system can recommend (or be recommended) immediately, before any retraining.

```
User tower called with user_id = <UNK>:
   Pretty good — declared interests + activity carry the signal.

Item tower called with item_id = <UNK>:
   Very good — text (MiniLM) + category do most of the work anyway.
   New items land within ~5–10% recall of fully-trained items.
```

---

## Hyperparameters

| Knob | Value | Why |
|---|---|---|
| `embed_dim` (output) | 64 | POC scale. Plenty for 4k items, FAISS instant. |
| `id_emb_dim` | 32 | Compact. |
| `category_emb_dim` | 16 | 12 categories — small space is fine. |
| `activity_emb_dim` | 8 | 3 levels — 8-d is generous. |
| `user_mlp_hidden` | 128 | Single hidden layer, ReLU, dropout 0.1. |
| `item_mlp_hidden` | 256 | Larger because input is larger (~434-d). |
| `batch_size` | 128 | 128 in-batch negatives. Bigger batch = more negatives = stronger signal but slower. |
| `optimizer` | Adam | `lr=1e-3`, default betas. |
| `epochs` | 10 | Loss plateaus around epoch 8 in practice. |
| `unk_dropout_rate` | 0.01 | 1% of IDs replaced with `<UNK>` per batch — enough for cold-start to work. |
| `random_seed` | 42 | All RNGs (numpy, torch, python). Reproducible. |

---

## Outputs (all in `artifacts/`)

| File | Shape / Content | Used by |
|---|---|---|
| `two_tower.pt` | torch state dict — both towers' weights | Cold-start tower forward pass at serving time. |
| `text_embeddings.npy` | `(num_items, 384)` float32 | Cached MiniLM features. Reused across training runs (only need to recompute when items change). |
| `user_embeddings.npy` | `(num_users, 64)` float32 | The `long_term_user_emb` lookup at serving time. |
| `item_embeddings.npy` | `(num_items, 64)` float32 | Source for FAISS index + the `recent_item` embeddings used in session blending. |
| `train_meta.json` | seed, hyperparams, final loss, runtime | Reproducibility record. |

---

## How to run (Phase 2, not yet implemented)

```bash
# 1. Cache item text embeddings (MiniLM, ~30 s on CPU)
python -m src.models.text_features

# 2. Train the two-tower model (~5 min on CPU)
python -m src.models.train_two_tower

# 3. Export the trained user + item embeddings
python -m src.models.export_embeddings
```

Each step writes to `artifacts/`. The next phase (`src/indexing/build_faiss.py`) reads `item_embeddings.npy` and builds the FAISS index.

---

## Sanity check after training

The training run is "good" when:

- Final mean batch loss < 1.0 (random would be ~`log(128) ≈ 4.85`).
- Recall@10 on the eval set > popularity baseline by a clear margin.
- For a "Travel-loving" user, the top-10 retrieved items are >70% from `Travel` + related categories.

If the loss stays >2.0 or recall is at-baseline, suspect:
- Item text features not actually being read in (check `text_embeddings.npy` shape).
- Bug in negative sampling (every item ranked equally — check the diagonal of the softmax).
- Learning rate too high (drop to 5e-4).
