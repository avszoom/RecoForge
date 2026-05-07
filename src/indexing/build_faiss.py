"""Build the FAISS item index from trained embeddings.

Reads:
    artifacts/item_embeddings.npy        (num_items, dim) float32, L2-normalized
    artifacts/item_id_to_row.json        {item_id: row_in_npy}

Writes (in artifacts/):
    item_index.faiss                     IndexFlatIP, dim = embedding dim
    item_mapping.json                    {dim, n_items, row_to_item_id}

This is the input to the recommender service. After training+export you
build the index once; cold-start additions later go through
`ItemIndex.add(...)` (see incremental_index.py) without rebuilding.

Usage:
    python -m src.indexing.build_faiss
    python -m src.indexing.build_faiss --item-emb artifacts/item_embeddings.npy --out artifacts/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.indexing.incremental_index import ItemIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_faiss")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the FAISS item index.")
    p.add_argument("--item-emb", type=Path, default=Path("artifacts/item_embeddings.npy"))
    p.add_argument("--mapping", type=Path, default=Path("artifacts/item_id_to_row.json"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--smoke-queries", type=int, default=3, help="run sanity searches at the end")
    args = p.parse_args()

    if not args.item_emb.exists():
        raise FileNotFoundError(f"{args.item_emb} not found — run `python -m src.models.export_embeddings` first")
    if not args.mapping.exists():
        raise FileNotFoundError(f"{args.mapping} not found — run `python -m src.models.export_embeddings` first")

    log.info("loading embeddings: %s", args.item_emb)
    item_emb = np.load(args.item_emb)
    if item_emb.dtype != np.float32:
        item_emb = item_emb.astype(np.float32)

    log.info("loading id mapping: %s", args.mapping)
    with args.mapping.open("r", encoding="utf-8") as f:
        item_id_to_row = json.load(f)

    if len(item_id_to_row) != item_emb.shape[0]:
        raise ValueError(
            f"mapping has {len(item_id_to_row)} entries; embeddings has {item_emb.shape[0]}"
        )

    # Sort item_ids by their .npy row so we insert into FAISS in the matching order.
    sorted_ids = sorted(item_id_to_row.items(), key=lambda kv: kv[1])
    item_ids = [iid for iid, _ in sorted_ids]
    rows = [r for _, r in sorted_ids]
    if rows != list(range(len(rows))):
        raise ValueError("item_id_to_row rows are non-contiguous")

    log.info("building index: dim=%d  n_items=%d", item_emb.shape[1], item_emb.shape[0])
    t0 = time.perf_counter()
    ix = ItemIndex(dim=item_emb.shape[1])
    ix.add_batch(item_ids, item_emb)
    log.info("index built in %.3fs", time.perf_counter() - t0)

    args.out.mkdir(parents=True, exist_ok=True)
    ix.save(args.out)
    log.info("saved → %s/%s + %s", args.out, ItemIndex.INDEX_FILENAME, ItemIndex.MAPPING_FILENAME)

    # ── smoke check: each item should be its own top-1 result (cosine 1.0) ──
    if args.smoke_queries > 0:
        log.info("smoke check: searching with %d items as queries", args.smoke_queries)
        sample_idx = np.linspace(0, item_emb.shape[0] - 1, args.smoke_queries, dtype=int)
        results = ix.search(item_emb[sample_idx], k=5)
        ok = True
        for q_row, hits in zip(sample_idx, results):
            top = hits[0]
            if top[0] != item_ids[q_row]:
                log.error("  query=%s but top-1=%s (score=%.4f) — mismatch!", item_ids[q_row], top[0], top[1])
                ok = False
                continue
            top_score = top[1]
            second = hits[1] if len(hits) > 1 else None
            log.info(
                "  query=%s → top1=%s (cos=%.4f)  top2=%s (cos=%.4f)",
                item_ids[q_row], top[0], top_score,
                second[0] if second else "—", second[1] if second else 0.0,
            )
            if abs(top_score - 1.0) > 1e-3:
                log.warning("  expected top-1 cosine ≈ 1.0; got %.4f", top_score)
                ok = False
        if not ok:
            raise RuntimeError("smoke check failed — see log above")
        log.info("smoke check OK")


if __name__ == "__main__":
    main()
