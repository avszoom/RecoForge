"""CLI: cold-start a new user — Phase 6.

Runs the trained user tower with user_id=<UNK> and the supplied profile,
extends artifacts/user_embeddings.npy + user_id_to_row.json, appends a
row to data/users.jsonl. The new user is immediately recommendable.

Usage:
    python -m src.serving.add_user \\
        --interests "AI Infrastructure" "Programming" \\
        --age-bucket 25-34 --location US --activity-level high --show-recs

    # explicit user_id
    python -m src.serving.add_user --user-id u_TEST_42 \\
        --interests Travel Food --age-bucket 18-24 --location EU
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.constants import ACTIVITY_LEVELS, AGE_BUCKETS, CATEGORIES, LOCATIONS
from src.serving.recommender import Recommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("add_user")


def main() -> None:
    p = argparse.ArgumentParser(description="Cold-start a new user.")
    p.add_argument("--interests", nargs="+", required=True, choices=list(CATEGORIES),
                   help="one or more interest categories")
    p.add_argument("--age-bucket", required=True, choices=list(AGE_BUCKETS))
    p.add_argument("--location", required=True, choices=list(LOCATIONS))
    p.add_argument("--activity-level", default="medium", choices=list(ACTIVITY_LEVELS))
    p.add_argument("--user-id", default=None, help="explicit id; auto-assigned if omitted")
    p.add_argument("--show-recs", action="store_true", help="print top-10 recs for the new user")
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--data", type=Path, default=Path("data"))
    args = p.parse_args()

    rec = Recommender(args.artifacts, args.data)
    user_id = rec.add_user(
        user_id=args.user_id,
        age_bucket=args.age_bucket,
        location=args.location,
        interests=args.interests,
        activity_level=args.activity_level,
    )
    print(f"\n  ✓ created user {user_id}  interests={args.interests}  activity={args.activity_level}")

    if args.show_recs:
        recs = rec.recommend(user_id, k=10, mode="adaptive")
        print(f"\n  Top 10 immediate recommendations for {user_id}:")
        print(f"    {'#':<3}{'score':<8}{'category':<22}{'sources':<22}title")
        print(f"    {'─'*3} {'─'*7} {'─'*21} {'─'*21} {'─'*60}")
        interests = set(args.interests)
        for r in recs:
            match = "★" if r.category in interests else " "
            sources = ",".join(r.sources)[:21]
            title_short = (r.title[:60] + "…") if len(r.title) > 60 else r.title
            print(f"    {r.rank:<3}{r.score:<8.3f}{r.category:<22}{sources:<22}{match} {title_short!r}")
        n_match = sum(1 for r in recs if r.category in interests)
        print(f"\n    ★ in declared interests: {n_match}/{len(recs)}")


if __name__ == "__main__":
    main()
