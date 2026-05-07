"""Lazy-loaded engine for cold-start tower inference.

NOTE: this module sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time. The
Recommender imports faiss-cpu (which loads libomp); when we later import
torch, a second libomp load triggers a segfault on macOS unless the
duplicate check is disabled. Setting it here — before torch is ever
imported — keeps the rest of the app blissfully unaware.

Used by `Recommender.add_user` and `Recommender.add_item` to produce a
64-d embedding for an entity that wasn't in the training set.

The `<UNK>` mechanic:
    Row 0 of both id-lookup tables (user_id, item_id) is reserved as an
    "unknown id" token. During training, ~1% of rows have their id
    replaced with row 0, so the MLP learns to lean on the rest of the
    features when the id signal is missing. At cold-start time, we just
    pass id=0 along with the entity's real features; the trained tower
    produces a sensible embedding in the same 64-d space as known users
    and items.

The model + MiniLM are loaded on first use, not at process startup, so
recommend() and on_click() callers don't pay the load cost.
"""

from __future__ import annotations

import os

# Must be set before any torch import. See module docstring.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

log = logging.getLogger("cold_start")


# Default text encoder. Must match what was used during training.
DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ColdStartEngine:
    """Wraps the trained TwoTower + frozen MiniLM behind a tiny inference API."""

    def __init__(self, artifacts_dir: Path | str):
        self.artifacts_dir = Path(artifacts_dir)
        self._loaded = False
        self._model = None                 # type: ignore[assignment]
        self._embedder = None              # type: ignore[assignment]
        self._cat_to_idx: dict[str, int] = {}
        self._act_to_idx: dict[str, int] = {}
        self._num_categories: int = 0

    # ── lazy load ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import torch                                              # heavy — import on demand
        from sentence_transformers import SentenceTransformer
        from src.models.two_tower import TwoTower, TwoTowerConfig

        ckpt_path = self.artifacts_dir / "two_tower.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"{ckpt_path} not found — Phase 2 hasn't been run "
                "(see src/models/README.md)"
            )

        log.info("loading trained TwoTower from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = TwoTowerConfig(**ckpt["config"])
        self._model = TwoTower(cfg)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._cat_to_idx = ckpt["cat_to_idx"]
        self._act_to_idx = ckpt["act_to_idx"]
        self._num_categories = cfg.num_categories

        log.info("loading text encoder: %s", DEFAULT_TEXT_MODEL)
        self._embedder = SentenceTransformer(DEFAULT_TEXT_MODEL)

        self._loaded = True

    # ── public API ────────────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.cfg.output_dim)

    def encode_text(self, text: str) -> np.ndarray:
        """Frozen MiniLM encoding of a title+body string. Returns 384-d L2-normalized."""
        self._ensure_loaded()
        emb = self._embedder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )[0]
        return emb.astype(np.float32)

    def compute_item_embedding(
        self, *,
        category: str,
        title: str,
        body: str,
        popularity: float = 0.0,
        freshness: float = 1.0,
    ) -> np.ndarray:
        """Run the item tower with item_id=<UNK>; return a 64-d L2-normalized vector."""
        import torch
        self._ensure_loaded()

        if category not in self._cat_to_idx:
            raise ValueError(f"unknown category: {category!r}")

        text_emb = self.encode_text(f"{title}. {body}")
        with torch.no_grad():
            emb = self._model.item_tower(
                item_id=torch.tensor([0], dtype=torch.long),                          # <UNK>
                category=torch.tensor([self._cat_to_idx[category]], dtype=torch.long),
                text_emb=torch.from_numpy(text_emb).unsqueeze(0).float(),
                popularity=torch.tensor([float(popularity)], dtype=torch.float),
                freshness=torch.tensor([float(freshness)], dtype=torch.float),
            )
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def compute_user_embedding(
        self, *,
        activity_level: str,
        interests: Iterable[str],
    ) -> np.ndarray:
        """Run the user tower with user_id=<UNK>; return a 64-d L2-normalized vector."""
        import torch
        self._ensure_loaded()

        if activity_level not in self._act_to_idx:
            raise ValueError(f"unknown activity_level: {activity_level!r}")
        interests_list = list(interests)
        for c in interests_list:
            if c not in self._cat_to_idx:
                raise ValueError(f"unknown interest: {c!r}")
        if not interests_list:
            raise ValueError("at least one interest is required for cold start")

        pref_mh = np.zeros((1, self._num_categories), dtype=np.float32)
        for c in interests_list:
            pref_mh[0, self._cat_to_idx[c]] = 1.0

        with torch.no_grad():
            emb = self._model.user_tower(
                user_id=torch.tensor([0], dtype=torch.long),                          # <UNK>
                activity=torch.tensor([self._act_to_idx[activity_level]], dtype=torch.long),
                preferred_multihot=torch.from_numpy(pref_mh),
            )
        return emb.squeeze(0).cpu().numpy().astype(np.float32)
