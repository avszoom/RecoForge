"""Phase 4 — long-term-only recommender.

The simplest end-to-end pipeline that produces real recommendations:

    user_id  →  long-term user_embedding  →  FAISS top-k  →  filter seen  →  k items

This is the version that uses ONLY the offline-trained vectors. No session
blending, no trending, no fresh, no ranker — those land in Phase 5.

CLI usage:
    python -m src.serving.recommender u_0042
    python -m src.serving.recommender u_0042 --k 20 --no-filter-seen
    python -m src.serving.recommender --random           # pick a random user

Programmatic usage:
    from src.serving.recommender import Recommender
    rec = Recommender()                                  # loads artifacts/ + data/
    items = rec.recommend("u_0042", k=10)                # list[Recommendation]

The Recommender exposes its internal state (`user_emb`, `index`,
`user_history`, `items_by_id`) so Phase 5 can plug in candidate generators
and ranking on top without re-loading anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.indexing.incremental_index import ItemIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("recommender")


# ─── output record ───────────────────────────────────────────────────────


@dataclass
class Recommendation:
    item_id: str
    score: float
    title: str
    category: str
    topic: str
    body: str
    popularity: float
    created_at: str
    rank: int = 0          # populated by the recommender, 1-indexed

    @property
    def title_short(self) -> str:
        return (self.title[:65] + "…") if len(self.title) > 65 else self.title


# ─── core ────────────────────────────────────────────────────────────────


class Recommender:
    """Long-term-only recommender. Stateless across requests; thread-safe for read."""

    def __init__(self, artifacts_dir: Path | str = "artifacts", data_dir: Path | str = "data"):
        artifacts_dir = Path(artifacts_dir)
        data_dir = Path(data_dir)

        # FAISS item index (Phase 3)
        self.index: ItemIndex = ItemIndex.load(artifacts_dir)
        log.info("loaded FAISS index: dim=%d  n_items=%d", self.index.dim, self.index.n_items)

        # Long-term user embeddings (Phase 2)
        self.user_emb: np.ndarray = np.load(artifacts_dir / "user_embeddings.npy")
        with (artifacts_dir / "user_id_to_row.json").open("r", encoding="utf-8") as f:
            self.user_id_to_row: dict[str, int] = json.load(f)
        log.info("loaded user embeddings: %s  num_users=%d", self.user_emb.shape, len(self.user_id_to_row))

        # Item metadata (for display + popularity/freshness lookups)
        self.items_by_id: dict[str, dict] = {}
        with (data_dir / "items.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    it = json.loads(line)
                    self.items_by_id[it["item_id"]] = it
        log.info("loaded item catalog: %d items", len(self.items_by_id))

        # User profile metadata (for display + interest-match diagnostics)
        self.users_by_id: dict[str, dict] = {}
        with (data_dir / "users.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    u = json.loads(line)
                    self.users_by_id[u["user_id"]] = u

        # User interaction history → seen-item filter
        self.user_history: dict[str, set[str]] = {}
        for fname in ("interactions.jsonl", "interactions_eval.jsonl"):
            path = data_dir / fname
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    self.user_history.setdefault(r["user_id"], set()).add(r["item_id"])
        log.info(
            "loaded interaction history: %d users with logged events",
            len(self.user_history),
        )

    # ── public API ────────────────────────────────────────────────────────

    def has_user(self, user_id: str) -> bool:
        return user_id in self.user_id_to_row

    def long_term_embedding(self, user_id: str) -> np.ndarray:
        """Return the trained long-term embedding for a known user.

        Raises KeyError for unknown users — Phase 6's add_user flow handles
        cold-start by running the user tower with <UNK> id; this method is
        intentionally strict so callers don't silently misuse it.
        """
        if user_id not in self.user_id_to_row:
            raise KeyError(f"unknown user_id: {user_id!r} — use add_user (Phase 6) for cold start")
        return self.user_emb[self.user_id_to_row[user_id]]

    def user_profile(self, user_id: str) -> dict | None:
        return self.users_by_id.get(user_id)

    def seen_items(self, user_id: str) -> set[str]:
        return self.user_history.get(user_id, set())

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        *,
        filter_seen: bool = True,
        overfetch: int = 50,
    ) -> list[Recommendation]:
        """Return the top-k items for a known user using long-term embeddings only.

        Overfetches `k * 3 + overfetch` candidates from FAISS so seen-item
        filtering rarely runs out of replacements. If a user has many seen
        items in their top neighborhood, increase `overfetch`.
        """
        u_vec = self.long_term_embedding(user_id)

        n_candidates = min(self.index.n_items, k * 3 + overfetch)
        hits = self.index.search(u_vec, k=n_candidates)[0]

        seen = self.seen_items(user_id) if filter_seen else set()
        recs: list[Recommendation] = []
        for item_id, score in hits:
            if item_id in seen:
                continue
            it = self.items_by_id.get(item_id)
            if it is None:
                continue
            recs.append(
                Recommendation(
                    item_id=it["item_id"],
                    score=score,
                    title=it["title"],
                    category=it["category"],
                    topic=it["topic"],
                    body=it["body"],
                    popularity=it["popularity_score"],
                    created_at=it["created_at"],
                )
            )
            if len(recs) >= k:
                break

        for i, r in enumerate(recs, start=1):
            r.rank = i
        return recs


# ─── CLI ─────────────────────────────────────────────────────────────────


def _print_user_block(rec: Recommender, user_id: str) -> None:
    profile = rec.user_profile(user_id)
    history = rec.seen_items(user_id)
    print(f"\nUser: {user_id}")
    if profile:
        print(f"  interests:        {profile.get('interests')}")
        print(f"  activity_level:   {profile.get('activity_level')}")
        print(f"  age / location:   {profile.get('age_bucket')} / {profile.get('location')}")
    print(f"  history:          {len(history)} logged interactions")


def _print_recs(recs: list[Recommendation], user_interests: list[str] | None) -> None:
    print(f"\nTop {len(recs)} recommendations (long-term only):")
    print(f"  {'#':<3}{'cos':<7}{'category':<22}title")
    print(f"  {'─'*3} {'─'*6} {'─'*21} {'─'*65}")
    for r in recs:
        match = "★" if user_interests and r.category in user_interests else " "
        print(f"  {r.rank:<3}{r.score:<7.3f}{r.category:<22}{match} {r.title_short!r}")
    if user_interests:
        n_match = sum(1 for r in recs if r.category in user_interests)
        print(f"\n  ★ = item is in user's declared interests   ({n_match}/{len(recs)} match)")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 — long-term-only recommender.")
    p.add_argument("user_id", nargs="?", help="user_id like 'u_0042'. Omit and use --random for a random user.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--no-filter-seen", action="store_true",
                   help="don't drop items the user has already interacted with")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--random", action="store_true",
                   help="pick a random user_id (useful for demos)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    recommender = Recommender(args.artifacts, args.data)

    user_id = args.user_id
    if not user_id:
        if not args.random:
            p.error("supply a user_id or pass --random")
        user_id = random.choice(list(recommender.user_id_to_row.keys()))
        log.info("picked random user: %s", user_id)
    if not recommender.has_user(user_id):
        p.error(f"unknown user_id: {user_id}")

    _print_user_block(recommender, user_id)
    profile = recommender.user_profile(user_id)
    interests = profile.get("interests") if profile else None
    recs = recommender.recommend(user_id, k=args.k, filter_seen=not args.no_filter_seen)
    _print_recs(recs, interests)


if __name__ == "__main__":
    main()
