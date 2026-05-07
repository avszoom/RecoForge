"""Per-user runtime state — recent clicks + session embedding cache.

This is the *online* state that exists outside the trained model. The
session embedding is computed on demand from the user's recent click
history (averaged item-vectors from the FAISS-indexed item embeddings),
not stored persistently — only the click history is persisted, since the
embedding can be recomputed cheaply from it.

State layout on disk: a single JSON file at `artifacts/user_state.json`,
keyed by user_id. Loaded into memory at process startup, saved after
every click. Tiny file even with thousands of users (kilobytes).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# Cap on how many recent clicks contribute to the session embedding.
# Long enough to dampen single-click noise, short enough that a real
# interest shift takes effect within a session.
RECENT_ITEMS_CAP: int = 20


@dataclass
class UserState:
    user_id: str
    recent_clicked_items: list[str] = field(default_factory=list)   # most-recent appended at end
    recent_categories: Counter = field(default_factory=Counter)
    n_clicks_total: int = 0

    # Cached session embedding — None means "needs recompute from recent_clicked_items".
    # Not persisted to disk; rebuilt from recent_clicked_items at recommend-time.
    session_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def record_click(self, item_id: str, category: str) -> None:
        """Append a click; cap history to RECENT_ITEMS_CAP; invalidate session cache."""
        self.recent_clicked_items.append(item_id)
        if len(self.recent_clicked_items) > RECENT_ITEMS_CAP:
            self.recent_clicked_items.pop(0)
        self.recent_categories[category] += 1
        self.n_clicks_total += 1
        self.session_embedding = None

    def clear(self) -> None:
        self.recent_clicked_items.clear()
        self.recent_categories.clear()
        self.session_embedding = None
        # n_clicks_total deliberately preserved — useful as an "all-time" stat for the UI.

    def to_json(self) -> dict:
        return {
            "user_id": self.user_id,
            "recent_clicked_items": list(self.recent_clicked_items),
            "recent_categories": dict(self.recent_categories),
            "n_clicks_total": self.n_clicks_total,
        }

    @classmethod
    def from_json(cls, data: dict) -> "UserState":
        return cls(
            user_id=data["user_id"],
            recent_clicked_items=list(data.get("recent_clicked_items", [])),
            recent_categories=Counter(data.get("recent_categories", {})),
            n_clicks_total=int(data.get("n_clicks_total", 0)),
        )


class UserStateStore:
    """Persistent dict[user_id, UserState] backed by a single JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.states: dict[str, UserState] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            for uid, data in payload.get("users", {}).items():
                self.states[uid] = UserState.from_json(data)

    def get_or_create(self, user_id: str) -> UserState:
        if user_id not in self.states:
            self.states[user_id] = UserState(user_id=user_id)
        return self.states[user_id]

    def get(self, user_id: str) -> Optional[UserState]:
        return self.states.get(user_id)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"users": {uid: st.to_json() for uid, st in self.states.items()}}
        # Atomic write: tmp file → rename, so a crashed write can't corrupt state.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp.replace(self.path)

    def reset(self, user_id: Optional[str] = None) -> None:
        """Clear state for one user (or all users if user_id is None)."""
        if user_id is None:
            self.states.clear()
        else:
            self.states.pop(user_id, None)
