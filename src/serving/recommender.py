"""Recommender — Phase 4 (long-term-only) + Phase 5 (adaptive).

Pipeline:
    user_id  →  long-term embedding  ─┐
                                       │  blend  ─→  final_user_embedding
    user state (recent clicks)        ─┘
                                       │
                  ┌─────────  generators (5 sources) ─────────┐
                  │   ann_long_term, similar_to_recent,        │
                  │   trending, fresh, category_interest       │
                  └────────────────────┬───────────────────────┘
                                       │
                                       ▼
                                  merge + rank
                                       │
                                       ▼
                              80/10/10 diversify
                                       │
                                       ▼
                            list[Recommendation]

CLI usage:
    python -m src.serving.recommender u_0042
    python -m src.serving.recommender u_0042 --mode long_term
    python -m src.serving.recommender u_0042 --click item_01234     # record + recompute
    python -m src.serving.recommender --random --seed 7

Programmatic usage:
    rec = Recommender()
    rec.recommend("u_0042", k=10)               # mode='adaptive' by default
    rec.on_click("u_0042", "item_01234")
    rec.recommend("u_0042")                     # next page reflects the click
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

from src.indexing.incremental_index import ItemIndex
from src.serving.candidate_generators import generate_all
from src.serving.ranker import MergedCandidate, diversify, merge, rank
from src.serving.user_state import UserState, UserStateStore

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
    rank: int = 0
    sources: list[str] = field(default_factory=list)            # ["ann", "trending", ...]
    source_scores: dict[str, float] = field(default_factory=dict)

    @property
    def title_short(self) -> str:
        return (self.title[:65] + "…") if len(self.title) > 65 else self.title


# ─── core ────────────────────────────────────────────────────────────────


Mode = Literal["adaptive", "long_term"]

# Session-blend weights as a function of the user's recent-click count.
# (long_term_weight, session_weight)
def _blend_weights(n_clicks: int) -> tuple[float, float]:
    if n_clicks < 3:    return (0.3, 0.7)        # cold-start: trust session more
    if n_clicks < 10:   return (0.5, 0.5)
    return (0.7, 0.3)                            # established: anchor on long-term


class Recommender:
    """Adaptive recommender with offline embeddings + online state."""

    def __init__(
        self,
        artifacts_dir: Path | str = "artifacts",
        data_dir: Path | str = "data",
        *,
        user_state_path: Path | str | None = None,
    ):
        artifacts_dir = Path(artifacts_dir)
        data_dir = Path(data_dir)

        # ── FAISS item index (Phase 3) ───────────────────────────────────
        self.index: ItemIndex = ItemIndex.load(artifacts_dir)
        log.info("loaded FAISS index: dim=%d  n_items=%d", self.index.dim, self.index.n_items)

        # ── Long-term embeddings (Phase 2) ───────────────────────────────
        self.user_emb: np.ndarray = np.load(artifacts_dir / "user_embeddings.npy")
        self.item_emb: np.ndarray = np.load(artifacts_dir / "item_embeddings.npy")
        with (artifacts_dir / "user_id_to_row.json").open("r", encoding="utf-8") as f:
            self.user_id_to_row: dict[str, int] = json.load(f)
        log.info("loaded user_emb=%s  item_emb=%s", self.user_emb.shape, self.item_emb.shape)

        # ── Catalogs ─────────────────────────────────────────────────────
        self.items_by_id: dict[str, dict] = {}
        self.items_by_category: dict[str, list[dict]] = defaultdict(list)
        with (data_dir / "items.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    it = json.loads(line)
                    self.items_by_id[it["item_id"]] = it
                    self.items_by_category[it["category"]].append(it)
        # Sort each category by popularity_score desc so category_interest can take .head().
        for cat, lst in self.items_by_category.items():
            lst.sort(key=lambda it: it["popularity_score"], reverse=True)

        self.users_by_id: dict[str, dict] = {}
        with (data_dir / "users.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    u = json.loads(line)
                    self.users_by_id[u["user_id"]] = u
        log.info("loaded items=%d  users=%d", len(self.items_by_id), len(self.users_by_id))

        # ── Interaction history → seen-item filter + trending counter ────
        self.user_history: dict[str, set[str]] = {}
        self.trending_counter: Counter = Counter()
        for fname in ("interactions.jsonl", "interactions_eval.jsonl", "online_interactions.jsonl"):
            path = data_dir / fname
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    self.user_history.setdefault(r["user_id"], set()).add(r["item_id"])
                    # Weight trending by event_weight so 'shares' matter more than 'views'.
                    if r["item_id"] in self.items_by_id:
                        self.trending_counter[r["item_id"]] += float(r.get("event_weight", 1.0))
        log.info(
            "loaded history: %d users · trending pool: %d items",
            len(self.user_history), len(self.trending_counter),
        )

        # ── Fresh items list (last 7 days, sorted by age asc) ────────────
        ref_time = datetime.now(timezone.utc)
        self.fresh_items_sorted: list[tuple[str, float]] = []
        for iid, item in self.items_by_id.items():
            try:
                age_days = (ref_time - datetime.fromisoformat(item["created_at"])).total_seconds() / 86400.0
            except ValueError:
                continue
            if 0.0 <= age_days <= 7.0:
                self.fresh_items_sorted.append((iid, age_days))
        self.fresh_items_sorted.sort(key=lambda x: x[1])
        log.info("fresh pool: %d items in last 7 days", len(self.fresh_items_sorted))

        # ── User state store (online clicks + session embedding cache) ───
        usp = Path(user_state_path) if user_state_path else artifacts_dir / "user_state.json"
        self.user_state: UserStateStore = UserStateStore(usp)
        self._online_log_path: Path = data_dir / "online_interactions.jsonl"
        log.info("user_state path: %s · loaded states: %d", usp, len(self.user_state.states))

    # ── lookups ──────────────────────────────────────────────────────────

    def has_user(self, user_id: str) -> bool:
        return user_id in self.user_id_to_row

    def long_term_embedding(self, user_id: str) -> np.ndarray:
        if user_id not in self.user_id_to_row:
            raise KeyError(f"unknown user_id: {user_id!r} — use add_user (Phase 6) for cold start")
        return self.user_emb[self.user_id_to_row[user_id]]

    def user_profile(self, user_id: str) -> dict | None:
        return self.users_by_id.get(user_id)

    def seen_items(self, user_id: str) -> set[str]:
        return self.user_history.get(user_id, set())

    # ── session embedding + dynamic blend ────────────────────────────────

    def _compute_session_embedding(self, state: UserState) -> np.ndarray | None:
        """Mean of recently clicked item embeddings, L2-normalized.

        Returns None if no clicks (or none that are in the FAISS index).
        Cached on the UserState — invalidated by record_click().
        """
        if state.session_embedding is not None:
            return state.session_embedding
        rows = [
            self.index.item_id_to_row[iid]
            for iid in state.recent_clicked_items
            if iid in self.index.item_id_to_row
        ]
        if not rows:
            return None
        avg = self.item_emb[rows].mean(axis=0)
        n = float(np.linalg.norm(avg))
        if n < 1e-9:
            return None
        avg = (avg / n).astype(np.float32)
        state.session_embedding = avg
        return avg

    def final_user_embedding(self, user_id: str) -> np.ndarray:
        """Blend long-term and session embeddings per the dynamic weight schedule."""
        long_term = self.long_term_embedding(user_id)
        state = self.user_state.get_or_create(user_id)
        session = self._compute_session_embedding(state)
        if session is None:
            return long_term
        w_long, w_session = _blend_weights(len(state.recent_clicked_items))
        blend = w_long * long_term + w_session * session
        n = float(np.linalg.norm(blend))
        return (blend / n).astype(np.float32) if n > 1e-9 else long_term

    # ── online click ─────────────────────────────────────────────────────

    def on_click(self, user_id: str, item_id: str, *, persist: bool = True) -> None:
        """Record a click: update UserState, trending, history; append to online log."""
        if not self.has_user(user_id):
            raise KeyError(f"unknown user_id: {user_id!r}")
        item = self.items_by_id.get(item_id)
        if item is None:
            raise KeyError(f"unknown item_id: {item_id!r}")

        state = self.user_state.get_or_create(user_id)
        state.record_click(item_id, item["category"])

        self.trending_counter[item_id] += 1.0           # click weight = 1.0
        self.user_history.setdefault(user_id, set()).add(item_id)

        if persist:
            self._online_log_path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "user_id": user_id,
                "item_id": item_id,
                "event_type": "click",
                "event_weight": 1.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with self._online_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            self.user_state.save()

    # ── recommend ────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        *,
        mode: Mode = "adaptive",
        filter_seen: bool = True,
    ) -> list[Recommendation]:
        if mode == "long_term":
            return self._recommend_long_term(user_id, k=k, filter_seen=filter_seen)
        if mode == "adaptive":
            return self._recommend_adaptive(user_id, k=k, filter_seen=filter_seen)
        raise ValueError(f"unknown mode: {mode!r}")

    def _recommend_long_term(self, user_id: str, k: int, *, filter_seen: bool) -> list[Recommendation]:
        u_vec = self.long_term_embedding(user_id)
        n_candidates = min(self.index.n_items, k * 3 + 50)
        hits = self.index.search(u_vec, k=n_candidates)[0]
        seen = self.seen_items(user_id) if filter_seen else set()

        recs: list[Recommendation] = []
        for item_id, score in hits:
            if item_id in seen:
                continue
            it = self.items_by_id.get(item_id)
            if it is None:
                continue
            recs.append(self._make_recommendation(it, score, sources=["ann"], source_scores={"ann": score}))
            if len(recs) >= k:
                break
        for i, r in enumerate(recs, start=1):
            r.rank = i
        return recs

    def _recommend_adaptive(self, user_id: str, k: int, *, filter_seen: bool) -> list[Recommendation]:
        # 1. Generate candidates from all 5 sources.
        candidates_per_source = generate_all(self, user_id)

        # 2. Merge dedupe.
        merged = merge(candidates_per_source)

        # 3. Score with the linear ranker.
        profile = self.user_profile(user_id)
        interests = set(profile.get("interests", []) if profile else [])
        scored = rank(merged, interests, self.items_by_id)

        # 4. 80/10/10 diversify with seen-item filter.
        seen = self.seen_items(user_id) if filter_seen else set()
        final = diversify(scored, seen, k=k)

        # 5. Wrap into Recommendation records.
        return [self._make_recommendation_from_merged(mc) for mc in final]

    # ── helpers ──────────────────────────────────────────────────────────

    def _make_recommendation(
        self, item: dict, score: float, *,
        sources: list[str], source_scores: dict[str, float],
    ) -> Recommendation:
        return Recommendation(
            item_id=item["item_id"],
            score=float(score),
            title=item["title"],
            category=item["category"],
            topic=item["topic"],
            body=item["body"],
            popularity=float(item["popularity_score"]),
            created_at=item["created_at"],
            sources=sources,
            source_scores=source_scores,
        )

    def _make_recommendation_from_merged(self, mc: MergedCandidate) -> Recommendation:
        return Recommendation(
            item_id=mc.item_id,
            score=mc.final_score,
            title=mc.title,
            category=mc.category,
            topic=mc.topic,
            body=mc.body,
            popularity=mc.popularity,
            created_at=mc.created_at,
            rank=mc.rank,
            sources=sorted(mc.source_scores.keys()),
            source_scores=dict(mc.source_scores),
        )


# ─── CLI ─────────────────────────────────────────────────────────────────


def _print_user_block(rec: Recommender, user_id: str) -> None:
    profile = rec.user_profile(user_id)
    history = rec.seen_items(user_id)
    state = rec.user_state.get(user_id)
    print(f"\nUser: {user_id}")
    if profile:
        print(f"  interests:        {profile.get('interests')}")
        print(f"  activity_level:   {profile.get('activity_level')}")
    print(f"  history (offline+online): {len(history)} interactions")
    if state and state.recent_clicked_items:
        print(f"  recent clicks ({len(state.recent_clicked_items)}): {state.recent_clicked_items[-5:]}")
        print(f"  recent categories: {dict(state.recent_categories)}")


def _print_recs(recs: list[Recommendation], user_interests: list[str] | None, mode: str) -> None:
    print(f"\nTop {len(recs)} recommendations · mode={mode}")
    print(f"  {'#':<3}{'score':<8}{'category':<22}{'sources':<22}title")
    print(f"  {'─'*3} {'─'*7} {'─'*21} {'─'*21} {'─'*60}")
    for r in recs:
        match = "★" if user_interests and r.category in user_interests else " "
        sources = ",".join(r.sources)[:21]
        print(f"  {r.rank:<3}{r.score:<8.3f}{r.category:<22}{sources:<22}{match} {r.title_short!r}")
    if user_interests:
        n_match = sum(1 for r in recs if r.category in user_interests)
        print(f"\n  ★ = item is in user's declared interests   ({n_match}/{len(recs)} match)")


def main() -> None:
    p = argparse.ArgumentParser(description="RecoForge recommender — Phase 4 + 5.")
    p.add_argument("user_id", nargs="?", help="user_id like 'u_0042'. Omit + use --random for a random user.")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--mode", choices=("adaptive", "long_term"), default="adaptive")
    p.add_argument("--no-filter-seen", action="store_true")
    p.add_argument("--click", default=None,
                   help="record a click on this item_id BEFORE producing recs (Phase 5 demo)")
    p.add_argument("--reset", action="store_true",
                   help="reset this user's online state (recent clicks, session embedding) before recommending")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--random", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    rec = Recommender(args.artifacts, args.data)

    user_id = args.user_id
    if not user_id:
        if not args.random:
            p.error("supply a user_id or pass --random")
        user_id = random.choice(list(rec.user_id_to_row.keys()))
        log.info("picked random user: %s", user_id)
    if not rec.has_user(user_id):
        p.error(f"unknown user_id: {user_id}")

    if args.reset:
        rec.user_state.reset(user_id)
        rec.user_state.save()
        log.info("reset online state for %s", user_id)

    if args.click:
        if args.click not in rec.items_by_id:
            p.error(f"unknown item_id: {args.click}")
        rec.on_click(user_id, args.click)
        log.info("recorded click: %s → %s (%s)", user_id, args.click, rec.items_by_id[args.click]["category"])

    _print_user_block(rec, user_id)
    profile = rec.user_profile(user_id)
    interests = profile.get("interests") if profile else None
    recs = rec.recommend(user_id, k=args.k, mode=args.mode, filter_seen=not args.no_filter_seen)
    _print_recs(recs, interests, args.mode)


if __name__ == "__main__":
    main()
