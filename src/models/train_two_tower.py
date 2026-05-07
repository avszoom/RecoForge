"""Two-tower training loop with in-batch sampled softmax.

THE LOSS, IN ENGLISH
====================
For each batch of 128 (user, item) interaction pairs:

    1. Forward both towers → 128 user vectors, 128 item vectors.
    2. Score matrix S[i, j] = dot(user_i, item_j) × exp(logit_scale)   shape (128, 128)
       Diagonal S[i, i] is the REAL pair from the interaction log.
       Off-diagonal S[i, j != i] is "user_i scored against another batch member's item"
       — these are our free, sampled negatives.
    3. Apply log_softmax row-wise → log P(item_j | user_i) for every (i, j).
    4. Pull out the diagonal: log P(item_i | user_i). This should approach 0 (i.e. probability 1)
       if the model perfectly puts the right item on top for each user.
    5. Loss = -mean( log_p_diagonal × event_weight )
       (event_weight makes a "share" count 4x more than a "view".)

Backprop pushes the model to *raise* diagonal scores and *lower* off-diagonal scores
across millions of (user, item) pairs over the course of training. After convergence,
similar users and items live near each other in the 64-d embedding space.

USAGE
=====
    python -m src.models.train_two_tower
    python -m src.models.train_two_tower --epochs 15 --batch-size 256 --lr 5e-4

Outputs (in artifacts/):
    two_tower.pt      torch state dict + the id-to-idx mappings
    train_meta.json   seed, hyperparams, loss curve, runtime
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.constants import ACTIVITY_LEVELS, CATEGORIES, EVENT_WEIGHTS
from src.models.two_tower import TwoTower, TwoTowerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_two_tower")


# ─── feature builders ────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _freshness(created_at_iso: str, ref: datetime, half_life_days: float = 14.0) -> float:
    """Exponential decay: brand-new = 1.0, 14d old ≈ 0.37, 30d old ≈ 0.12."""
    age_days = max(0.0, (ref - datetime.fromisoformat(created_at_iso)).total_seconds() / 86400.0)
    return float(math.exp(-age_days / half_life_days))


class FeatureStore:
    """Holds the precomputed per-user and per-item feature tensors.

    Indexing convention everywhere: row 0 is reserved for <UNK>.
    A real user/item with ID like "u_0042" maps to row 43 (= 42 + 1).
    """

    def __init__(self, users: list[dict], items: list[dict], text_emb: np.ndarray, ref_time: datetime):
        # ── mappings ──────────────────────────────────────────────────────
        self.user_id_to_idx = {u["user_id"]: i + 1 for i, u in enumerate(users)}
        self.item_id_to_idx = {it["item_id"]: i + 1 for i, it in enumerate(items)}
        self.cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
        self.act_to_idx = {a: i for i, a in enumerate(ACTIVITY_LEVELS)}
        self.num_categories = len(CATEGORIES)

        # ── user features (rows 1..num_users; row 0 is <UNK>, left zero) ──
        n_users = len(users)
        self.user_activity = torch.zeros(n_users + 1, dtype=torch.long)
        self.user_pref_multihot = torch.zeros(n_users + 1, self.num_categories, dtype=torch.float)
        for u in users:
            row = self.user_id_to_idx[u["user_id"]]
            self.user_activity[row] = self.act_to_idx[u["activity_level"]]
            for c in u["interests"]:
                self.user_pref_multihot[row, self.cat_to_idx[c]] = 1.0

        # ── item features (rows 1..num_items) ─────────────────────────────
        n_items = len(items)
        self.item_category = torch.zeros(n_items + 1, dtype=torch.long)
        self.item_text = torch.zeros(n_items + 1, text_emb.shape[1], dtype=torch.float)
        self.item_popularity = torch.zeros(n_items + 1, dtype=torch.float)
        self.item_freshness = torch.zeros(n_items + 1, dtype=torch.float)
        for it in items:
            row = self.item_id_to_idx[it["item_id"]]
            self.item_category[row] = self.cat_to_idx[it["category"]]
            self.item_text[row] = torch.from_numpy(text_emb[row - 1])    # text_emb is 0-indexed by item order
            self.item_popularity[row] = float(it["popularity_score"])
            self.item_freshness[row] = _freshness(it["created_at"], ref_time)

    # convenience batch lookups
    def gather_users(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return idx, self.user_activity[idx], self.user_pref_multihot[idx]

    def gather_items(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            idx,
            self.item_category[idx],
            self.item_text[idx],
            self.item_popularity[idx],
            self.item_freshness[idx],
        )


class InteractionDataset(Dataset):
    """One row per training interaction: (user_idx, item_idx, event_weight)."""

    def __init__(self, interactions: list[dict], store: FeatureStore):
        self.user_idx = torch.tensor(
            [store.user_id_to_idx[r["user_id"]] for r in interactions], dtype=torch.long
        )
        self.item_idx = torch.tensor(
            [store.item_id_to_idx[r["item_id"]] for r in interactions], dtype=torch.long
        )
        self.event_w = torch.tensor(
            [float(r["event_weight"]) for r in interactions], dtype=torch.float
        )

    def __len__(self) -> int:
        return self.user_idx.size(0)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_idx[i], self.item_idx[i], self.event_w[i]


# ─── training step ───────────────────────────────────────────────────────


def _apply_unk_dropout(idx: torch.Tensor, rate: float) -> torch.Tensor:
    """Replace `rate` fraction of the IDs with row 0 (<UNK>).

    This teaches the towers to handle unseen users/items at serving time
    by leaning on the rest of their features.
    """
    if rate <= 0.0:
        return idx
    mask = torch.rand_like(idx, dtype=torch.float) < rate
    return torch.where(mask, torch.zeros_like(idx), idx)


def train_one_epoch(
    model: TwoTower,
    loader: DataLoader,
    store: FeatureStore,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    unk_dropout: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    diag_sum = 0.0       # mean cosine sim of true positive pairs (should grow)
    offdiag_sum = 0.0    # mean cosine sim of in-batch negatives  (should stay low)
    n_batches = 0

    for user_idx, item_idx, event_w in loader:
        # ── cold-start regularization ────────────────────────────────────
        user_idx = _apply_unk_dropout(user_idx.to(device), unk_dropout)
        item_idx = _apply_unk_dropout(item_idx.to(device), unk_dropout)
        event_w = event_w.to(device)

        # ── pull features ────────────────────────────────────────────────
        u_id, u_act, u_pref = store.gather_users(user_idx)
        i_id, i_cat, i_text, i_pop, i_fresh = store.gather_items(item_idx)

        # ── forward ──────────────────────────────────────────────────────
        u_emb = model.user_tower(u_id, u_act, u_pref)               # (B, 64)
        i_emb = model.item_tower(i_id, i_cat, i_text, i_pop, i_fresh)  # (B, 64)

        # ── THE LOSS ─────────────────────────────────────────────────────
        # 1. Score every user against every item in the batch.
        scores = model.score_matrix(u_emb, i_emb)            # (B, B)

        # 2. Convert each row into a probability distribution via log_softmax.
        #    Row i: log P(item_j | user_i) for j in 0..B-1.
        log_probs = F.log_softmax(scores, dim=1)             # (B, B)

        # 3. The diagonal is the true positive: log P(item_i | user_i).
        log_p_pos = log_probs.diagonal()                     # (B,)

        # 4. Cross-entropy loss, weighted by event importance.
        loss = -(log_p_pos * event_w).sum() / event_w.sum()

        # ── backward ─────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── diagnostics (no grads) ───────────────────────────────────────
        with torch.no_grad():
            B = scores.size(0)
            cos_sim = u_emb @ i_emb.T                        # raw cosine, no scaling
            diag = cos_sim.diagonal().mean().item()
            off = (cos_sim.sum() - cos_sim.diagonal().sum()) / max(1, B * (B - 1))
            diag_sum += diag
            offdiag_sum += off.item()

        total_loss += loss.item() * event_w.sum().item()
        total_weight += event_w.sum().item()
        n_batches += 1

    return {
        "loss": total_loss / max(1.0, total_weight),
        "mean_diag_cos": diag_sum / max(1, n_batches),
        "mean_offdiag_cos": offdiag_sum / max(1, n_batches),
    }


# ─── main ────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Train the RecoForge two-tower model.")
    p.add_argument("--users", type=Path, default=Path("data/users.jsonl"))
    p.add_argument("--items", type=Path, default=Path("data/items.jsonl"))
    p.add_argument("--interactions", type=Path, default=Path("data/interactions.jsonl"))
    p.add_argument("--text-emb", type=Path, default=Path("artifacts/text_embeddings.npy"))
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--unk-dropout", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu | cuda | mps (auto-detect by default)")
    args = p.parse_args()

    # ── reproducibility ──────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device_str = args.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    log.info("device: %s", device)

    # ── load data ────────────────────────────────────────────────────────
    log.info("[1/5] loading data")
    users = _load_jsonl(args.users)
    items = _load_jsonl(args.items)
    interactions = _load_jsonl(args.interactions)
    log.info("       users=%d  items=%d  train_interactions=%d", len(users), len(items), len(interactions))

    log.info("[2/5] loading text embeddings")
    text_emb = np.load(args.text_emb)
    if text_emb.shape[0] != len(items):
        raise ValueError(f"text_embeddings rows ({text_emb.shape[0]}) != items ({len(items)})")

    log.info("[3/5] building feature store")
    ref_time = datetime.now(timezone.utc)
    store = FeatureStore(users, items, text_emb, ref_time)
    # move feature tensors onto device once (small enough that this is cheap)
    for attr in ("user_activity", "user_pref_multihot", "item_category", "item_text", "item_popularity", "item_freshness"):
        setattr(store, attr, getattr(store, attr).to(device))

    # ── build model ──────────────────────────────────────────────────────
    log.info("[4/5] building model")
    cfg = TwoTowerConfig(
        num_users=len(users),
        num_items=len(items),
        num_categories=len(CATEGORIES),
        text_dim=text_emb.shape[1],
    )
    model = TwoTower(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("       params=%d  user_tower_out=%d  scale_init=%.2f", n_params, cfg.output_dim, math.exp(cfg.init_logit_scale))

    # ── data loader ──────────────────────────────────────────────────────
    dataset = InteractionDataset(interactions, store)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,           # in-batch sampled softmax wants square (B, B) matrices
        num_workers=0,
    )
    log.info("       batches/epoch=%d  batch_size=%d", len(loader), args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── train ────────────────────────────────────────────────────────────
    log.info("[5/5] training")
    history: list[dict] = []
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.perf_counter()
        stats = train_one_epoch(
            model, loader, store, optimizer, device,
            unk_dropout=args.unk_dropout,
        )
        stats["epoch"] = epoch
        stats["seconds"] = round(time.perf_counter() - ep_t0, 2)
        stats["logit_scale"] = round(model.logit_scale.exp().item(), 3)
        history.append(stats)
        log.info(
            "  epoch %2d/%d  loss=%.4f  diag=%.4f  off-diag=%.4f  scale=%.2f  (%.1fs)",
            epoch, args.epochs, stats["loss"], stats["mean_diag_cos"],
            stats["mean_offdiag_cos"], stats["logit_scale"], stats["seconds"],
        )
    total_time = time.perf_counter() - t0
    log.info("training done in %.1fs", total_time)

    # ── save ─────────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "user_id_to_idx": store.user_id_to_idx,
        "item_id_to_idx": store.item_id_to_idx,
        "cat_to_idx": store.cat_to_idx,
        "act_to_idx": store.act_to_idx,
    }
    torch.save(payload, args.out / "two_tower.pt")

    meta = {
        "seed": args.seed,
        "device": str(device),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "hyperparams": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "unk_dropout": args.unk_dropout,
        },
        "config": asdict(cfg),
        "history": history,
        "final_loss": history[-1]["loss"],
        "params": n_params,
        "total_seconds": round(total_time, 2),
    }
    with (args.out / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("saved → %s", args.out / "two_tower.pt")
    log.info("       %s", args.out / "train_meta.json")

    # ── sanity ───────────────────────────────────────────────────────────
    final = history[-1]
    if final["mean_diag_cos"] - final["mean_offdiag_cos"] < 0.05:
        log.warning("SANITY: diagonal vs off-diagonal cosine gap is tiny — model may not be learning.")
    if final["loss"] > math.log(args.batch_size):
        log.warning("SANITY: final loss above random-baseline (%.2f). Investigate.", math.log(args.batch_size))


if __name__ == "__main__":
    main()
