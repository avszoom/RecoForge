"""Run all baselines + the trained model over the held-out eval split.

Produces `artifacts/eval_report.json` with one row per strategy. The
Streamlit Evaluation page reads this file and renders a comparison table.

Strategies:
    1. popularity              global top-k by popularity_score, ignoring user
    2. category                user's preferred-category items by popularity
    3. two_tower_long_term     mode='long_term' — trained user emb + FAISS
    4. two_tower_adaptive      session-replay: replays each user's eval
                                 clicks in chronological order, predicts
                                 the next click given the session-blended
                                 embedding built from prior clicks

Methodology:
    Each held-out interaction (user, item, ts) is one evaluation query.
    We ask the strategy "what would you have recommended for this user?"
    and check whether `item` appears in the top-k. Metrics: recall@k,
    MRR@k, NDCG@k. filter_seen is OFF for fair comparison — strategies
    that filter would have an artificial advantage.

Usage:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --k-values 5,10,20
    python -m src.evaluation.evaluate --eval data/interactions_eval.jsonl --max-interactions 5000
"""

from __future__ import annotations

# Match recommender.py's libomp env-var setup so this module can be run as a
# standalone script without needing scripts/run_app.sh.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from src.evaluation.metrics import mrr, ndcg_at_k, recall_at_k
from src.serving.recommender import Recommender


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")


# ─── helpers ─────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _compute_metrics(
    pairs: list[tuple[str, list[str]]], k_values: list[int]
) -> dict[str, float]:
    out: dict[str, float] = {"n_eval_pairs": float(len(pairs))}
    for k in k_values:
        out[f"recall@{k}"] = round(recall_at_k(pairs, k), 5)
        out[f"mrr@{k}"] = round(mrr(pairs, k), 5)
        out[f"ndcg@{k}"] = round(ndcg_at_k(pairs, k), 5)
    return out


# ─── baselines ───────────────────────────────────────────────────────────


def evaluate_popularity(
    eval_interactions: list[dict],
    items_by_id: dict,
    k_max: int,
    k_values: list[int],
) -> dict:
    """Recommend the top-k items by popularity_score, same list for every user."""
    sorted_items = sorted(
        items_by_id.values(), key=lambda it: it["popularity_score"], reverse=True
    )
    top = [it["item_id"] for it in sorted_items[:k_max]]
    pairs = [(r["item_id"], top) for r in eval_interactions]
    return _compute_metrics(pairs, k_values)


def evaluate_category(
    eval_interactions: list[dict],
    items_by_category: dict,
    users_by_id: dict,
    k_max: int,
    k_values: list[int],
) -> dict:
    """Per-user: pool items from declared interests, sort by popularity, take top-k."""
    # Cache per-user top-k since the same user may appear many times in eval.
    cache: dict[str, list[str]] = {}

    def top_for(uid: str) -> list[str]:
        if uid in cache:
            return cache[uid]
        u = users_by_id.get(uid)
        if not u or not u.get("interests"):
            cache[uid] = []
            return []
        pool: list[tuple[str, float]] = []
        for cat in u["interests"]:
            for it in items_by_category.get(cat, []):
                pool.append((it["item_id"], float(it["popularity_score"])))
        pool.sort(key=lambda x: x[1], reverse=True)
        cache[uid] = [iid for iid, _ in pool[:k_max]]
        return cache[uid]

    pairs = [(r["item_id"], top_for(r["user_id"])) for r in eval_interactions]
    return _compute_metrics(pairs, k_values)


def evaluate_two_tower_long_term(
    rec: Recommender,
    eval_interactions: list[dict],
    k_max: int,
    k_values: list[int],
) -> dict:
    """mode='long_term' — uses only the trained user embedding, no session blend."""
    pairs: list[tuple[str, list[str]]] = []
    for r in eval_interactions:
        if not rec.has_user(r["user_id"]):
            pairs.append((r["item_id"], []))
            continue
        recs = rec.recommend(r["user_id"], k=k_max, mode="long_term", filter_seen=False)
        pairs.append((r["item_id"], [rr.item_id for rr in recs]))
    return _compute_metrics(pairs, k_values)


def evaluate_two_tower_adaptive(
    rec: Recommender,
    eval_interactions: list[dict],
    k_max: int,
    k_values: list[int],
) -> dict:
    """Session-replay evaluation.

    For each user: sort their eval interactions by timestamp, then for each
    one in order, predict it using `mode='adaptive'` with the session built
    from all prior eval clicks. After predicting, record the click so the
    next prediction sees it in the session.

    The first interaction for each user has an empty session, so the
    adaptive blend collapses to the long-term embedding for that prediction.
    """
    by_user: dict[str, list[dict]] = defaultdict(list)
    for r in eval_interactions:
        by_user[r["user_id"]].append(r)
    for uid in by_user:
        by_user[uid].sort(key=lambda r: r["timestamp"])

    pairs: list[tuple[str, list[str]]] = []
    for uid, ints in by_user.items():
        if not rec.has_user(uid):
            continue
        rec.user_state.reset(uid)
        for r in ints:
            if r["item_id"] not in rec.items_by_id:
                continue
            recs = rec.recommend(uid, k=k_max, mode="adaptive", filter_seen=False)
            pairs.append((r["item_id"], [rr.item_id for rr in recs]))
            # Feed this click into the session for the NEXT prediction.
            rec.on_click(uid, r["item_id"], persist=False)
        rec.user_state.reset(uid)        # cleanup so next user starts fresh
    return _compute_metrics(pairs, k_values)


# ─── main ────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate baselines + two-tower over the held-out eval split.")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--eval", type=Path, default=Path("data/interactions_eval.jsonl"))
    p.add_argument("--out", type=Path, default=Path("artifacts/eval_report.json"))
    p.add_argument("--k-values", default="5,10",
                   help="comma-separated list of k values for recall@k / mrr@k / ndcg@k (default 5,10)")
    p.add_argument("--max-interactions", type=int, default=0,
                   help="cap eval interactions for speed during dev (0 = use all)")
    args = p.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    k_max = max(k_values)
    log.info("k values: %s (k_max=%d)", k_values, k_max)

    log.info("loading recommender")
    rec = Recommender(args.artifacts, args.data)

    log.info("loading eval interactions: %s", args.eval)
    eval_interactions = _load_jsonl(args.eval)
    if args.max_interactions and args.max_interactions < len(eval_interactions):
        log.info("subsampling eval interactions: %d → %d", len(eval_interactions), args.max_interactions)
        eval_interactions = eval_interactions[: args.max_interactions]
    log.info("evaluating against %d interactions", len(eval_interactions))

    strategies = [
        ("popularity",
         lambda: evaluate_popularity(eval_interactions, rec.items_by_id, k_max, k_values)),
        ("category",
         lambda: evaluate_category(eval_interactions, rec.items_by_category, rec.users_by_id, k_max, k_values)),
        ("two_tower_long_term",
         lambda: evaluate_two_tower_long_term(rec, eval_interactions, k_max, k_values)),
        ("two_tower_adaptive",
         lambda: evaluate_two_tower_adaptive(rec, eval_interactions, k_max, k_values)),
    ]

    results: dict[str, dict] = {}
    for name, fn in strategies:
        log.info("→ %s", name)
        t0 = time.perf_counter()
        results[name] = fn()
        results[name]["seconds"] = round(time.perf_counter() - t0, 2)
        log.info(
            "    recall@%d=%.4f  mrr@%d=%.4f  ndcg@%d=%.4f  (%.1fs)",
            k_max, results[name][f"recall@{k_max}"],
            k_max, results[name][f"mrr@{k_max}"],
            k_max, results[name][f"ndcg@{k_max}"],
            results[name]["seconds"],
        )

    report = {
        "n_eval_interactions": len(eval_interactions),
        "k_values": k_values,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("wrote → %s", args.out)

    # ── compact comparison table on stdout ──────────────────────────────
    print()
    headers = ["strategy", *(f"recall@{k}" for k in k_values),
               *(f"mrr@{k}" for k in k_values),
               *(f"ndcg@{k}" for k in k_values), "seconds"]
    print("  ".join(f"{h:>20s}" for h in headers))
    for name, r in results.items():
        row = [name] + [f"{r.get(c, 0):.4f}" if isinstance(r.get(c, 0), float) else str(r.get(c, ""))
                        for c in headers[1:]]
        print("  ".join(f"{c:>20s}" for c in row))


if __name__ == "__main__":
    main()
