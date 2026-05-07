"""Export trained user + item embeddings as numpy arrays.

Run after training. Reads `artifacts/two_tower.pt` and runs every user and
every item through their tower once. Output `.npy` files are what the
serving layer (FAISS index, session blending) consumes — the model itself
is only loaded again for cold-start (new user / new item) inference.

Usage:
    python -m src.models.export_embeddings

Outputs (in artifacts/):
    user_embeddings.npy        shape: (num_users, output_dim), float32, L2-normalized
    item_embeddings.npy        shape: (num_items, output_dim), float32, L2-normalized
    user_id_to_row.json        {user_id: row_index_in_npy}
    item_id_to_row.json        {item_id: row_index_in_npy}

Note: the .npy files are 0-indexed by the order of users.jsonl / items.jsonl
(not by the +1 offset used internally for the <UNK> row). The mappings
make this explicit so the serving layer doesn't have to guess.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data.constants import CATEGORIES
from src.models.train_two_tower import FeatureStore, _load_jsonl
from src.models.two_tower import TwoTower, TwoTowerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("export_embeddings")


@torch.no_grad()
def run_user_tower(model: TwoTower, store: FeatureStore, n_users: int, batch: int = 256) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out = np.zeros((n_users, model.cfg.output_dim), dtype=np.float32)
    # rows 1..n_users in the lookup tables correspond to .npy rows 0..n_users-1
    for start in range(0, n_users, batch):
        end = min(n_users, start + batch)
        idx = torch.arange(start + 1, end + 1, device=device)
        u_id, u_act, u_pref = store.gather_users(idx)
        emb = model.user_tower(u_id, u_act, u_pref)
        out[start:end] = emb.cpu().numpy()
    return out


@torch.no_grad()
def run_item_tower(model: TwoTower, store: FeatureStore, n_items: int, batch: int = 256) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out = np.zeros((n_items, model.cfg.output_dim), dtype=np.float32)
    for start in range(0, n_items, batch):
        end = min(n_items, start + batch)
        idx = torch.arange(start + 1, end + 1, device=device)
        i_id, i_cat, i_text, i_pop, i_fresh = store.gather_items(idx)
        emb = model.item_tower(i_id, i_cat, i_text, i_pop, i_fresh)
        out[start:end] = emb.cpu().numpy()
    return out


# ─── retrieval quality probe ─────────────────────────────────────────────

def retrieval_probe(
    user_emb: np.ndarray, item_emb: np.ndarray, users: list[dict], items: list[dict],
    eval_interactions: list[dict], k: int = 10,
) -> dict[str, float]:
    """Cheap top-k retrieval check on the held-out eval set.

    For each (user, item) eval pair, run the user's vector against ALL items,
    take the top-k, and check whether the held-out item is in there (Recall@k)
    and whether it is the top-1 (HitRate@1). Also report category-match rate
    (does the top-1 share the user's declared interests?).
    """
    user_id_to_row = {u["user_id"]: i for i, u in enumerate(users)}
    item_id_to_row = {it["item_id"]: i for i, it in enumerate(items)}
    user_interests = {u["user_id"]: set(u["interests"]) for u in users}
    item_cat = {it["item_id"]: it["category"] for it in items}
    item_id_by_row = {i: it["item_id"] for i, it in enumerate(items)}

    hits_at_k = 0
    hits_at_1 = 0
    cat_match_top1 = 0
    n = 0

    for r in eval_interactions:
        if r["user_id"] not in user_id_to_row or r["item_id"] not in item_id_to_row:
            continue
        u_row = user_id_to_row[r["user_id"]]
        true_item_row = item_id_to_row[r["item_id"]]

        scores = item_emb @ user_emb[u_row]                         # (num_items,)
        top_k_rows = np.argpartition(-scores, k)[:k]
        top_k_rows = top_k_rows[np.argsort(-scores[top_k_rows])]
        top_1_row = top_k_rows[0]

        if true_item_row in top_k_rows:
            hits_at_k += 1
        if true_item_row == top_1_row:
            hits_at_1 += 1

        if item_cat[item_id_by_row[top_1_row]] in user_interests[r["user_id"]]:
            cat_match_top1 += 1
        n += 1

    return {
        "n_eval_pairs": n,
        f"recall@{k}": round(hits_at_k / max(1, n), 4),
        "hit@1": round(hits_at_1 / max(1, n), 4),
        "cat_match@1": round(cat_match_top1 / max(1, n), 4),
    }


# ─── main ────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Export trained two-tower embeddings.")
    p.add_argument("--users", type=Path, default=Path("data/users.jsonl"))
    p.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    p.add_argument("--eval-interactions", type=Path, default=Path("data/interactions_eval.jsonl"))
    p.add_argument("--text-emb", type=Path, default=Path("artifacts/text_embeddings.npy"))
    p.add_argument("--ckpt", type=Path, default=Path("artifacts/two_tower.pt"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    args = p.parse_args()

    log.info("loading checkpoint: %s", args.ckpt)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = TwoTowerConfig(**payload["config"])
    model = TwoTower(cfg)
    model.load_state_dict(payload["state_dict"])
    device_str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = model.to(device)
    log.info("loaded model (params=%d) on %s", sum(p.numel() for p in model.parameters()), device)

    log.info("loading data + features")
    users = _load_jsonl(args.users)
    items = _load_jsonl(args.items)
    text_emb = np.load(args.text_emb)
    store = FeatureStore(users, items, text_emb, datetime.now(timezone.utc))
    for attr in ("user_activity", "user_pref_multihot", "item_category", "item_text", "item_popularity", "item_freshness"):
        setattr(store, attr, getattr(store, attr).to(device))

    log.info("running towers")
    t0 = time.perf_counter()
    user_arr = run_user_tower(model, store, len(users))
    item_arr = run_item_tower(model, store, len(items))
    log.info("done in %.1fs (users: %s, items: %s)", time.perf_counter() - t0, user_arr.shape, item_arr.shape)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "user_embeddings.npy", user_arr)
    np.save(args.out / "item_embeddings.npy", item_arr)

    user_id_to_row = {u["user_id"]: i for i, u in enumerate(users)}
    item_id_to_row = {it["item_id"]: i for i, it in enumerate(items)}
    with (args.out / "user_id_to_row.json").open("w", encoding="utf-8") as f:
        json.dump(user_id_to_row, f)
    with (args.out / "item_id_to_row.json").open("w", encoding="utf-8") as f:
        json.dump(item_id_to_row, f)

    # ── retrieval quality probe ──────────────────────────────────────────
    log.info("retrieval probe on held-out eval set")
    eval_int = _load_jsonl(args.eval_interactions) if args.eval_interactions.exists() else []
    if eval_int:
        # Subsample for speed (full set has ~20k pairs; 2k is enough for a noise-free read).
        sample = eval_int[: 2000]
        report = retrieval_probe(user_arr, item_arr, users, items, sample, k=10)
        log.info("       %s", report)
        with (args.out / "retrieval_probe.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
