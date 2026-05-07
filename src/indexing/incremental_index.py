"""FAISS IndexFlatIP wrapper with a string-id ↔ row mapping.

Why a wrapper:
    - FAISS works in integer row positions, but the rest of the system
      uses string item_ids ("item_00123"). The wrapper keeps both in sync.
    - Cold-start adds at runtime (Phase 6) need a single API; calling
      faiss directly leaves the row→id mapping out of sync on every call.
    - Save/load is opinionated: one .faiss binary plus one .json mapping
      file in a single directory.

Index choice:
    `IndexFlatIP` — exact inner-product search, no training, no
    quantization. Vectors are L2-normalized at insertion time so inner
    product equals cosine similarity. At our scale (4k items) this is
    sub-millisecond per query and the simplest thing that works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np


class ItemIndex:
    """FAISS index keyed by string item_ids."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.row_to_item_id: list[str] = []
        self.item_id_to_row: dict[str, int] = {}

    @property
    def n_items(self) -> int:
        return int(self.index.ntotal)

    # ── insertion ────────────────────────────────────────────────────────

    def add_batch(self, item_ids: Sequence[str], vectors: np.ndarray) -> None:
        """Insert many items at once. Vectors will be L2-normalized."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (N, {self.dim}); got {vectors.shape}")
        if len(item_ids) != vectors.shape[0]:
            raise ValueError(f"item_ids ({len(item_ids)}) != vectors ({vectors.shape[0]})")
        for iid in item_ids:
            if iid in self.item_id_to_row:
                raise ValueError(f"item_id already in index: {iid!r}")

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-9, None)

        start = self.n_items
        self.index.add(vectors)
        for offset, iid in enumerate(item_ids):
            self.row_to_item_id.append(iid)
            self.item_id_to_row[iid] = start + offset

    def add(self, item_id: str, vector: np.ndarray) -> int:
        """Insert a single item; return its row index."""
        v = vector if vector.ndim == 2 else vector.reshape(1, -1)
        self.add_batch([item_id], v)
        return self.item_id_to_row[item_id]

    # ── search ───────────────────────────────────────────────────────────

    def search(self, queries: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        """Return top-k (item_id, cosine_score) lists per query.

        Single query is fine — pass a 1-d vector and you'll get a list with
        one inner list back.
        """
        if self.n_items == 0:
            return [[] for _ in range(queries.shape[0] if queries.ndim == 2 else 1)]

        q = queries if queries.ndim == 2 else queries.reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"expected query dim {self.dim}; got {q.shape[1]}")
        q = np.ascontiguousarray(q, dtype=np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.clip(norms, 1e-9, None)

        k = min(k, self.n_items)
        scores, idxs = self.index.search(q, k)

        results: list[list[tuple[str, float]]] = []
        for batch_idxs, batch_scores in zip(idxs, scores):
            row: list[tuple[str, float]] = []
            for idx, score in zip(batch_idxs, batch_scores):
                if idx < 0 or idx >= len(self.row_to_item_id):
                    continue
                row.append((self.row_to_item_id[idx], float(score)))
            results.append(row)
        return results

    # ── persistence ──────────────────────────────────────────────────────

    INDEX_FILENAME = "item_index.faiss"
    MAPPING_FILENAME = "item_mapping.json"

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / self.INDEX_FILENAME))
        with (directory / self.MAPPING_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "n_items": self.n_items,
                    "row_to_item_id": self.row_to_item_id,
                },
                f,
            )

    @classmethod
    def load(cls, directory: Path) -> "ItemIndex":
        with (directory / cls.MAPPING_FILENAME).open("r", encoding="utf-8") as f:
            meta = json.load(f)
        ix = cls(dim=int(meta["dim"]))
        ix.index = faiss.read_index(str(directory / cls.INDEX_FILENAME))
        ix.row_to_item_id = list(meta["row_to_item_id"])
        ix.item_id_to_row = {iid: row for row, iid in enumerate(ix.row_to_item_id)}
        if ix.index.ntotal != len(ix.row_to_item_id):
            raise RuntimeError(
                f"index/mapping out of sync: ntotal={ix.index.ntotal} mapping={len(ix.row_to_item_id)}"
            )
        return ix
