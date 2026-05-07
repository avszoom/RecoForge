"""CLI: cold-start a new item — Phase 6.

Runs the trained item tower with item_id=<UNK>, freshness=1.0, popularity=0.0,
inserts the new vector into FAISS, appends to data/items.jsonl, persists
all updated artifacts. The new item is immediately retrievable.

Usage:
    python -m src.serving.add_item --category Travel \\
        --title "Why Lisbon is the perfect weekend escape" \\
        --body "Three days, walkable food, cheap flights..."

    # also recommend a user some recs to verify the new item shows up
    python -m src.serving.add_item --category "AI Infrastructure" \\
        --title "..." --body "..." --recs-for u_0007
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.constants import CATEGORIES
from src.serving.recommender import Recommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("add_item")


def main() -> None:
    p = argparse.ArgumentParser(description="Cold-start a new item.")
    p.add_argument("--category", required=True, choices=list(CATEGORIES))
    p.add_argument("--title", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--topic", default="", help="free-text topic; defaults to lowercase category")
    p.add_argument("--creator-id", default=None)
    p.add_argument("--item-id", default=None, help="explicit id; auto-assigned if omitted")
    p.add_argument("--recs-for", default=None,
                   help="user_id to query after adding — verifies the new item is reachable")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--data", type=Path, default=Path("data"))
    args = p.parse_args()

    rec = Recommender(args.artifacts, args.data)
    item_id = rec.add_item(
        item_id=args.item_id,
        category=args.category,
        title=args.title,
        body=args.body,
        topic=args.topic,
        creator_id=args.creator_id,
    )
    print(f"\n  ✓ created item {item_id}  category={args.category}  title={args.title!r}")

    if args.recs_for:
        if not rec.has_user(args.recs_for):
            log.error("unknown user_id: %s", args.recs_for)
            return
        recs = rec.recommend(args.recs_for, k=20, mode="adaptive")
        new_in_top = next((r for r in recs if r.item_id == item_id), None)
        print(f"\n  Recs for {args.recs_for} (top 20, new item highlighted):")
        for r in recs:
            mark = "*** NEW ***" if r.item_id == item_id else "           "
            sources = ",".join(r.sources)[:18]
            title_short = (r.title[:60] + "…") if len(r.title) > 60 else r.title
            print(f"    {mark}  rank={r.rank:<3} score={r.score:.3f} cat={r.category:<22} src={sources:<19} {title_short!r}")
        if new_in_top:
            print(f"\n  → new item appeared at rank {new_in_top.rank}")
        else:
            print(f"\n  → new item not in top 20 for this user (OK if their interests don't match {args.category})")


if __name__ == "__main__":
    main()
