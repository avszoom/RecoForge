"""Precompute frozen MiniLM embeddings for every item's title + body.

Run once before training. The 384-d vector for each item is cached as a
fixed feature; the item tower learns to project it into the 64-d space
without ever backproping into MiniLM. This makes training fast (no
forward through a 22M-param transformer per batch) and stable (we don't
have the data to fine-tune MiniLM well).

Usage:
    python -m src.models.text_features
    python -m src.models.text_features --items data/items.jsonl --out artifacts/

Outputs (in artifacts/):
    text_embeddings.npy        shape: (num_items, 384), float32, L2-normalized
    text_embeddings_meta.json  {item_id: row_index, ...} so callers can look up
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("text_features")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_items(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode_items(items: list[dict], model_name: str = DEFAULT_MODEL, batch_size: int = 64) -> np.ndarray:
    log.info("loading embedding model: %s", model_name)
    t0 = time.perf_counter()
    model = SentenceTransformer(model_name)
    log.info("model loaded in %.1fs (dim=%d)", time.perf_counter() - t0, model.get_sentence_embedding_dimension())

    texts = [f"{it['title']}. {it['body']}" for it in items]
    log.info("encoding %d items (batch_size=%d)", len(texts), batch_size)
    t0 = time.perf_counter()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    log.info("encoded %d items in %.1fs (shape=%s)", len(texts), time.perf_counter() - t0, embs.shape)
    return embs


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute frozen MiniLM features for items.")
    p.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    if not args.items.exists():
        raise FileNotFoundError(f"{args.items} not found — run `python -m src.data.generate_dataset` first")

    items = load_items(args.items)
    log.info("loaded %d items from %s", len(items), args.items)

    embs = encode_items(items, model_name=args.model, batch_size=args.batch_size)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "text_embeddings.npy", embs)

    item_id_to_row = {it["item_id"]: i for i, it in enumerate(items)}
    with (args.out / "text_embeddings_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"model": args.model, "num_items": len(items), "dim": int(embs.shape[1]), "item_id_to_row": item_id_to_row},
            f,
        )

    log.info("saved → %s (%.1f MB)", args.out / "text_embeddings.npy", embs.nbytes / 1e6)


if __name__ == "__main__":
    main()
