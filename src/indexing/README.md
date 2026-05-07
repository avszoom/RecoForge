# FAISS index — Phase 3

Wraps the trained item embeddings in a FAISS `IndexFlatIP` so the serving layer can find nearest items per user vector in milliseconds.

> **Status:** done. Build it once after Phase 2; it's then read by the serving layer and extended in place by Phase 6 cold-start.

---

## Why `IndexFlatIP`

| Property | Why it matters here |
|---|---|
| **Inner-product** scoring | Item + user embeddings are already L2-normalized, so dot product = cosine similarity. No separate `IndexFlatL2` math. |
| **Exact** search | Only 4,000 items. No need for IVF / HNSW / PQ approximations. |
| **No training step** | Index is built immediately from the .npy file — there's no codebook to learn. |
| **`add()` is O(1)** | Cold-start (Phase 6) appends new items at runtime via `index.add(vec)` — no rebuild. |
| **~1 ms** per top-10 query | At our scale, the bottleneck is anywhere but here. |

When item count climbs past ~100k, swap in `IndexIVFFlat` or `IndexHNSWFlat`. Below that, flat is the right answer.

---

## Files

```
src/indexing/
├── incremental_index.py    ItemIndex wrapper class (used at serving time)
└── build_faiss.py          one-shot CLI to build the index from item_embeddings.npy
```

---

## `ItemIndex` API

```python
from src.indexing.incremental_index import ItemIndex

ix = ItemIndex(dim=64)

# Bulk insert (used by build_faiss.py)
ix.add_batch(item_ids=["item_00000", ...], vectors=item_embeddings)

# Single insert (used by Phase 6 add_item)
ix.add(item_id="item_05000", vector=cold_start_emb)   # returns row index

# Top-k search — query may be 1-d or (B, dim)
results = ix.search(user_emb, k=10)
# → [[("item_00723", 0.877), ("item_02769", 0.868), ...]]

# Persistence
ix.save(Path("artifacts"))   # writes item_index.faiss + item_mapping.json
ItemIndex.load(Path("artifacts"))
```

The wrapper keeps two things in lockstep:
- the FAISS `IndexFlatIP` of vectors
- a `row_to_item_id: list[str]` and the inverse `item_id_to_row: dict[str, int]`

Vectors are L2-normalized at insertion (defensive — even if the caller forgets). Queries are L2-normalized too.

---

## How to build the index

```bash
python -m src.indexing.build_faiss
```

Reads `artifacts/item_embeddings.npy` and `artifacts/item_id_to_row.json` (both produced by `src/models/export_embeddings.py`), writes:

```
artifacts/
├── item_index.faiss      ~1 MB     IndexFlatIP, dim 64, 4000 items
└── item_mapping.json     ~80 KB    {dim, n_items, row_to_item_id: [...]}
```

The CLI also runs a self-search smoke check: query the index with each item's own embedding and verify it comes back as top-1 with cosine ≈ 1.0.

```
smoke check: searching with 3 items as queries
  query=item_00000 → top1=item_00000 (cos=1.0000)  top2=item_01398 (cos=0.9288)
  query=item_01999 → top1=item_01999 (cos=1.0000)  top2=item_02675 (cos=0.9447)
  query=item_03999 → top1=item_03999 (cos=1.0000)  top2=item_02448 (cos=0.9648)
smoke check OK
```

The `top2` cosines (0.93–0.96) reveal the within-category clustering documented in `src/models/README.md`'s known-limitations section.

---

## Tests

`tests/test_indexing.py` (6 tests):

- self-top-1: every inserted vector must come back as its own top-1 match.
- 1-d query input: shape compatibility (the wrapper auto-reshapes).
- save/load round-trip: results identical before and after.
- cold-start `add()` after `load()`: a new item inserted into a loaded index is searchable in the same call.
- duplicate-id rejection: `add()` raises if the id is already present.
- dim-mismatch rejection: `add_batch()` raises if vectors don't match `dim`.
