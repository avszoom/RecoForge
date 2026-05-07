"""ItemIndex tests: insert, search, save/load round-trip, cold-start add."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.indexing.incremental_index import ItemIndex


def _random_unit_vectors(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def test_add_batch_and_search_self_top1() -> None:
    """Each inserted vector should be its own top-1 result (cos = 1.0)."""
    n, d = 100, 16
    vecs = _random_unit_vectors(n, d, seed=1)
    ids = [f"item_{i:03d}" for i in range(n)]

    ix = ItemIndex(dim=d)
    ix.add_batch(ids, vecs)

    assert ix.n_items == n
    results = ix.search(vecs, k=3)
    for q_row, hits in enumerate(results):
        assert hits[0][0] == ids[q_row]
        assert pytest.approx(hits[0][1], rel=1e-4, abs=1e-4) == 1.0


def test_search_single_query_1d_input() -> None:
    """A 1-d query vector should still return a list-of-lists shape."""
    ix = ItemIndex(dim=4)
    ix.add_batch(["a", "b", "c"], _random_unit_vectors(3, 4, seed=2))

    results = ix.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert len(results) == 1
    assert len(results[0]) == 2


def test_save_load_round_trip(tmp_path: Path) -> None:
    n, d = 50, 8
    vecs = _random_unit_vectors(n, d, seed=3)
    ids = [f"item_{i:02d}" for i in range(n)]

    ix = ItemIndex(dim=d)
    ix.add_batch(ids, vecs)
    ix.save(tmp_path)

    loaded = ItemIndex.load(tmp_path)
    assert loaded.n_items == n
    assert loaded.row_to_item_id == ids

    # search results should be identical between original and reloaded index
    q = vecs[:5]
    a = ix.search(q, k=3)
    b = loaded.search(q, k=3)
    assert a == b


def test_cold_start_add_after_load(tmp_path: Path) -> None:
    """Insert a new item via .add() into a loaded index — should appear in subsequent searches."""
    d = 8
    vecs = _random_unit_vectors(20, d, seed=4)
    ids = [f"item_{i:02d}" for i in range(20)]
    ix = ItemIndex(dim=d)
    ix.add_batch(ids, vecs)
    ix.save(tmp_path)

    loaded = ItemIndex.load(tmp_path)
    new_vec = _random_unit_vectors(1, d, seed=5)[0]
    new_row = loaded.add("item_NEW", new_vec)
    assert new_row == 20
    assert loaded.n_items == 21
    assert loaded.item_id_to_row["item_NEW"] == 20

    # Searching with the new item's own vector should put it on top.
    results = loaded.search(new_vec, k=1)
    assert results[0][0][0] == "item_NEW"
    assert pytest.approx(results[0][0][1], abs=1e-4) == 1.0


def test_duplicate_id_rejected() -> None:
    ix = ItemIndex(dim=4)
    vecs = _random_unit_vectors(2, 4, seed=6)
    ix.add_batch(["a", "b"], vecs)
    with pytest.raises(ValueError, match="already in index"):
        ix.add("a", _random_unit_vectors(1, 4, seed=7)[0])


def test_dim_mismatch_rejected() -> None:
    ix = ItemIndex(dim=8)
    bad = _random_unit_vectors(1, 7, seed=8)
    with pytest.raises(ValueError, match="expected"):
        ix.add_batch(["x"], bad)
