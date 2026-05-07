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

---

## Known limitations (v1)

The trained model learns category routing **perfectly** (`cat_match@1 = 100%` on held-out pairs) but is weak at picking the exact item within a category (`recall@10 ≈ 1.35%` vs random 0.25%, so ~5× random). Final loss after 10 epochs is 3.81 — meaningfully below the random baseline of `log(128) = 4.85`, but well above what's achievable on real data. Two root causes:

### 1. Templated text looks too similar to MiniLM

The synthetic dataset uses templated titles like *"Reducing {topic}: {modifier}"*. All "AI Infrastructure" titles share most of their words, so MiniLM places them in nearly the same region of its 384-d space. The item tower can tell *AI Infra* from *Travel* but cannot tell AI-Infra item #5 from item #12.

> **This disappears with real (non-templated) text.** Production titles have varied vocabulary, length, and style, and MiniLM separates them well.

### 2. In-batch sampled softmax has a false-negative problem

Each batch gives one positive and 127 in-batch negatives. When user A and user B both like Travel, and the batch has `(A → travel_X)` and `(B → travel_Y)`, the loss says *"for user A, rank travel_Y LOW"* — but A would actually like travel_Y too. The training signal contradicts itself for within-category items, and the model effectively gives up trying to push within-category items toward each other in the embedding space.

This manifests as: training for too many epochs makes within-category recall *worse*, not better. We saw `recall@10` drop from 1.35% (10 epochs) to 0.6% (25 epochs).

### Why this is acceptable for v1

The RecoForge demo is *"click Travel → see more Travel."* That only needs **shelf-level routing**, which the model nails 100% of the time. Within-shelf ordering comes from other parts of the system, not from the model's fingerprints:

- **Session blending** averages recent item embeddings — coarse fingerprints are fine as long as they're in the right cluster.
- **Trending / fresh / category candidate generators** (Phase 5) handle within-category ranking via popularity and recency, not embedding similarity.
- **The ranking layer** (Phase 5) combines all of these signals.

### Diagnostic numbers (10-epoch run, seed=42)

| Metric | Value | Random baseline | Interpretation |
|---|---:|---:|---|
| Final loss | 3.81 | 4.85 | ~2.8× better than random |
| Mean diag cosine | 0.51 | 0.00 | True (user, item) pairs are similar |
| Mean off-diag cosine | 0.25 | 0.00 | In-batch negatives have residual category overlap |
| Diag − off-diag gap | **0.26** | 0.00 | What drives retrieval quality |
| `cat_match@1` | **100.0%** | ~25%* | Top-1 always in user's interests |
| `recall@10` | 1.35% | 0.25% | 5.4× random |
| `hit@1` | 0.10% | 0.025% | 4× random |

*Random `cat_match@1` ≈ user's interest count / 12 categories ≈ 2/12 ≈ 17%.

---

## V2 fixes (when this matters)

In rough order of effort vs payoff:

### 1. Decoupled negative sampling — cheapest, biggest single win

Replace in-batch negatives with **random items drawn from the full catalog**. Each positive gets ~256 random negatives sampled uniformly. Random items rarely overlap with the user's interests, so the false-negative problem disappears.

- **Cost:** ~20 LOC change to the training loop. Add a `random_negatives_per_positive` argument; sample item indices uniformly per batch; build the score matrix as `user_emb @ random_item_emb.T`.
- **Trade-off:** random negatives are easy. The model learns coarse distinctions (Travel ≠ Cooking) but doesn't have to work at fine ones. Recall@10 typically jumps 3–10×; recall@1 improves modestly.

### 2. Sampled-softmax bias correction — one-line fix

Popular items appear in many batches and get marked "negative" disproportionately often. Subtract `log P(item)` from each logit before the softmax — the standard sampled-softmax bias correction.

- **Cost:** one extra term in the loss + a one-time pass to count empirical item frequencies on the training set.
- **Impact:** prevents the model from over-penalizing popular items. Especially useful when paired with random/uniform negatives.

```python
# Before: loss is unfairly harsh on popular items
log_probs = F.log_softmax(scores, dim=1)

# After: subtract log P(item) — popular items get a "discount"
log_probs = F.log_softmax(scores - log_p_item.unsqueeze(0), dim=1)
```

### 3. Hard-negative mining — most work, biggest ceiling

Random negatives are too easy. Periodically run the current model, find items it ranks high for a user but the user didn't engage with, and use those as next-round negatives. Forces within-category discrimination.

- **Cost:** ~80 LOC, two-stage training pipeline. Need to retrain after each mining pass.
- **Risk:** hard negatives can be too hard early in training and collapse the model. Standard mitigation: start with random negatives and gradually mix in harder ones (curriculum).
- **Impact:** the only fix that materially improves *within-category* discrimination. Production two-tower systems almost always use some form of this.

### Other limitations worth noting

- **Frozen MiniLM** is general-purpose. Fine-tuning it on the engagement signal would help, but we don't have the data scale to do that responsibly.
- **No creator-side features.** `creator_id` is in the items file but ignored by the item tower. Adding a creator embedding could capture "users who like creator X also like ..." patterns.
- **1% `<UNK>` dropout** is a guess from CLIP-style cold-start papers. Production systems often go 5–10%, especially when the cold-start path is heavily used.
- **No temporal features beyond freshness.** Hour-of-day and day-of-week patterns are absent from both the dataset and the model.
- **Single-stage training.** Industry systems often pretrain on weak labels (impressions / clicks) then fine-tune on strong labels (saves / shares). We treat all events equally weighted by `event_weight`.

### When to actually invest in V2

For the **demo**: never. Category routing is enough.

For a **real recommender**:
- Start with V2.1 (decoupled negatives) — quickest win, fixes the obvious bug.
- Add V2.2 (bias correction) if popularity skew is bad.
- Invest in V2.3 (hard-negative mining) only after the dataset is real and large enough to support it. Synthetic data + hard negatives = wasted effort.
