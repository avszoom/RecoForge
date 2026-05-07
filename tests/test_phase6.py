"""Phase 6 — cold-start tests for add_user and add_item.

Cold-start runs the trained two-tower model with id=<UNK> and the new
entity's real features, then wires the resulting embedding into the
running system. These tests verify both the in-memory state changes
(arrays grow, FAISS index updates, catalogs reflect the new entity) and
the persistence path (files on disk are updated when persist=True).

The fixture uses persist=False to avoid mutating the canonical
data/artifacts. The persistence test uses a tmp copy.

NOTE: torch is imported BEFORE the Recommender (which loads faiss). On
macOS Apple Silicon, faiss + torch in the same process segfault inside
load_state_dict unless torch's libomp wins the load order.
"""

from __future__ import annotations

# IMPORTANT: torch must be imported before any faiss-loading import.
import torch  # noqa: F401  (load order)

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.serving.recommender import Recommender


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"


def _required_paths_exist() -> bool:
    return all([
        (ARTIFACTS / "two_tower.pt").exists(),
        (ARTIFACTS / "item_index.faiss").exists(),
        (ARTIFACTS / "item_embeddings.npy").exists(),
        (ARTIFACTS / "user_embeddings.npy").exists(),
        (DATA / "items.jsonl").exists(),
        (DATA / "users.jsonl").exists(),
    ])


@pytest.fixture(scope="function")
def recommender(tmp_path: Path) -> Recommender:
    if not _required_paths_exist():
        pytest.skip("artifacts/ + data/ not fully populated — run pipeline through Phase 3 first")
    return Recommender(ARTIFACTS, DATA, user_state_path=tmp_path / "state.json")


# ─── add_user (in-memory) ────────────────────────────────────────────────


def test_add_user_extends_arrays_in_memory(recommender: Recommender) -> None:
    n_before = recommender.user_emb.shape[0]
    new_uid = recommender.add_user(
        interests=["AI Infrastructure"],
        age_bucket="25-34", location="US", activity_level="high",
        persist=False,
    )
    assert new_uid in recommender.user_id_to_row
    assert recommender.user_id_to_row[new_uid] == n_before
    assert recommender.user_emb.shape[0] == n_before + 1
    assert recommender.user_emb.shape[1] == recommender.index.dim
    assert recommender.users_by_id[new_uid]["interests"] == ["AI Infrastructure"]
    # L2-normalized
    norm = float(np.linalg.norm(recommender.user_emb[n_before]))
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_add_user_rejects_invalid_inputs(recommender: Recommender) -> None:
    with pytest.raises(ValueError, match="age_bucket"):
        recommender.add_user(interests=["Travel"], age_bucket="0-10", location="US", persist=False)
    with pytest.raises(ValueError, match="location"):
        recommender.add_user(interests=["Travel"], age_bucket="25-34", location="MARS", persist=False)
    with pytest.raises(ValueError, match="activity_level"):
        recommender.add_user(
            interests=["Travel"], age_bucket="25-34", location="US",
            activity_level="extreme", persist=False,
        )
    with pytest.raises(ValueError, match="interest"):
        recommender.add_user(interests=["Underwater Basket Weaving"], age_bucket="25-34",
                             location="US", persist=False)
    with pytest.raises(ValueError, match="interests must be non-empty"):
        recommender.add_user(interests=[], age_bucket="25-34", location="US", persist=False)


def test_add_user_duplicate_id_rejected(recommender: Recommender) -> None:
    existing = next(iter(recommender.user_id_to_row))
    with pytest.raises(ValueError, match="already exists"):
        recommender.add_user(
            user_id=existing, interests=["Travel"], age_bucket="25-34",
            location="US", persist=False,
        )


def test_cold_start_user_gets_sensible_recs(recommender: Recommender) -> None:
    """A brand-new user with one declared interest should get top-10 mostly in that category."""
    new_uid = recommender.add_user(
        interests=["AI Infrastructure"],
        age_bucket="25-34", location="US", activity_level="high",
        persist=False,
    )
    recs = recommender.recommend(new_uid, k=10, mode="adaptive")
    in_interest = sum(1 for r in recs if r.category == "AI Infrastructure")
    assert in_interest >= 8, f"only {in_interest}/10 in declared interest"


# ─── add_item (in-memory) ────────────────────────────────────────────────


def test_add_item_extends_arrays_and_index(recommender: Recommender) -> None:
    n_before = recommender.item_emb.shape[0]
    n_index_before = recommender.index.n_items
    new_iid = recommender.add_item(
        category="Travel",
        title="A weekend in Porto: cheap flights, river views",
        body="Three days, walkable streets, food that ruins you for home.",
        persist=False,
    )
    assert new_iid in recommender.items_by_id
    assert recommender.item_emb.shape[0] == n_before + 1
    assert recommender.index.n_items == n_index_before + 1
    assert recommender.index.item_id_to_row[new_iid] == n_index_before
    # New item is in the fresh pool at age 0.
    assert recommender.fresh_items_sorted[0][0] == new_iid
    assert recommender.fresh_items_sorted[0][1] == 0.0


def test_add_item_rejects_invalid_category(recommender: Recommender) -> None:
    with pytest.raises(ValueError, match="category"):
        recommender.add_item(category="MARS", title="t", body="b", persist=False)
    with pytest.raises(ValueError, match="title and body are required"):
        recommender.add_item(category="Travel", title="", body="b", persist=False)


def test_cold_start_item_reachable_via_index(recommender: Recommender) -> None:
    new_iid = recommender.add_item(
        category="Travel",
        title="An off-season escape to the Faroe Islands",
        body="Wild cliffs, no crowds, surprising food scene, easy flights from Copenhagen.",
        persist=False,
    )
    # Search using the new item's own vector — must come back as top-1.
    new_row = recommender.index.item_id_to_row[new_iid]
    self_query = recommender.item_emb[new_row]
    hits = recommender.index.search(self_query, k=1)[0]
    assert hits[0][0] == new_iid
    assert pytest.approx(hits[0][1], abs=1e-4) == 1.0


def test_cold_start_item_appears_for_matching_user(recommender: Recommender) -> None:
    """A new Travel item should show up in a Travel-loving user's top-N adaptive recs."""
    new_iid = recommender.add_item(
        category="Travel",
        title="A weekend in Porto: cheap flights, river views",
        body="Three days, walkable streets, food that ruins you for home.",
        persist=False,
    )
    travel_users = [
        u["user_id"] for u in recommender.users_by_id.values()
        if u.get("interests") == ["Travel"]
    ]
    if not travel_users:
        pytest.skip("no single-interest Travel users in dataset")

    found_at = None
    for uid in travel_users[:5]:
        recommender.user_state.reset(uid)
        recs = recommender.recommend(uid, k=20, mode="adaptive")
        match = next((r for r in recs if r.item_id == new_iid), None)
        if match:
            found_at = match.rank
            break
    assert found_at is not None, "new Travel item never appeared in any Travel user's top-20"
    assert found_at <= 20


# ─── persistence (tmp copy of artifacts/data) ────────────────────────────


def _copy_artifacts(tmp: Path) -> tuple[Path, Path]:
    """Copy artifacts/ and data/ to tmp so the persist test doesn't pollute real data."""
    art = tmp / "artifacts"
    dat = tmp / "data"
    shutil.copytree(ARTIFACTS, art)
    shutil.copytree(DATA, dat)
    return art, dat


def test_add_user_persists_to_disk(tmp_path: Path) -> None:
    if not _required_paths_exist():
        pytest.skip("pipeline not run")
    art, dat = _copy_artifacts(tmp_path)
    rec = Recommender(art, dat, user_state_path=tmp_path / "state.json")

    new_uid = rec.add_user(
        interests=["Travel"], age_bucket="25-34", location="US",
        activity_level="medium", persist=True,
    )

    # users.jsonl should have a new last line containing this user_id.
    last_line = (dat / "users.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    record = json.loads(last_line)
    assert record["user_id"] == new_uid
    # Updated arrays were saved.
    user_emb_disk = np.load(art / "user_embeddings.npy")
    assert user_emb_disk.shape[0] == rec.user_emb.shape[0]
    with (art / "user_id_to_row.json").open("r") as f:
        idx_disk = json.load(f)
    assert new_uid in idx_disk


def test_add_item_persists_to_disk(tmp_path: Path) -> None:
    if not _required_paths_exist():
        pytest.skip("pipeline not run")
    art, dat = _copy_artifacts(tmp_path)
    rec = Recommender(art, dat, user_state_path=tmp_path / "state.json")

    new_iid = rec.add_item(
        category="Travel",
        title="A weekend in Porto",
        body="Cheap flights, walkable streets, easy food.",
        persist=True,
    )

    last_line = (dat / "items.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    record = json.loads(last_line)
    assert record["item_id"] == new_iid
    # Item embedding persisted.
    item_emb_disk = np.load(art / "item_embeddings.npy")
    assert item_emb_disk.shape[0] == rec.item_emb.shape[0]
    # FAISS index persisted.
    assert (art / "item_index.faiss").exists()
    with (art / "item_id_to_row.json").open("r") as f:
        idx_disk = json.load(f)
    assert new_iid in idx_disk
