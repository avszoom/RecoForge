"""RecoForge — Phase 7 Streamlit UI.

Four pages:
    1. Recommendations            — top-k for the chosen user, click-to-adapt
    2. Add new item               — Phase 6 cold-start with verification
    3. User state debugger        — long_term vs adaptive side-by-side
    4. Evaluation                 — Phase 8 results placeholder

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

# IMPORTANT: torch must be imported BEFORE faiss to keep load_state_dict
# stable when both libraries' libomp are present in the process.
# See src/serving/README.md → "Notes on the macOS faiss + torch combo".
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401  (load order)

import random
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Make project root importable when streamlit is launched from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.constants import ACTIVITY_LEVELS, AGE_BUCKETS, CATEGORIES, LOCATIONS
from src.serving.recommender import Recommendation, Recommender


ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"


# ─── page config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RecoForge",
    page_icon="⚡",
    layout="wide",
)


# ─── caching ─────────────────────────────────────────────────────────────


def _required_artifacts_present() -> tuple[bool, list[str]]:
    required = [
        ARTIFACTS / "item_index.faiss",
        ARTIFACTS / "user_embeddings.npy",
        ARTIFACTS / "item_embeddings.npy",
        ARTIFACTS / "two_tower.pt",
        DATA / "items.jsonl",
        DATA / "users.jsonl",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    return (not missing), missing


@st.cache_resource(show_spinner="Loading recommender (first call only — ~1s)…")
def get_recommender() -> Recommender:
    return Recommender(ARTIFACTS, DATA)


# ─── shared widgets ──────────────────────────────────────────────────────


SOURCE_COLORS = {
    "ann":      "blue",
    "recent":   "orange",
    "trending": "green",
    "fresh":    "violet",
    "category": "gray",
}


def _source_badges(sources: list[str]) -> str:
    """Return a markdown string with one colored badge per source."""
    return " ".join(f":{SOURCE_COLORS.get(s, 'gray')}-background[{s}]" for s in sources)


def _user_profile_card(rec: Recommender, user_id: str) -> None:
    profile = rec.user_profile(user_id)
    state = rec.user_state.get(user_id)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### `{user_id}`")
        if profile:
            st.markdown(
                f"**Interests:** {', '.join(profile['interests'])}  ·  "
                f"**Activity:** {profile['activity_level']}  ·  "
                f"**Location:** {profile['location']}  ·  "
                f"**Age:** {profile['age_bucket']}"
            )
        n_seen = len(rec.seen_items(user_id))
        st.caption(f"{n_seen} interactions in offline+online history")
    with col2:
        if state and state.recent_clicked_items:
            st.markdown("**Recent clicks (this session):**")
            for iid in reversed(state.recent_clicked_items[-5:]):
                item = rec.items_by_id.get(iid, {})
                cat = item.get("category", "?")
                title = item.get("title", iid)
                st.caption(f"• `[{cat}]` {title[:55]}")
            cats_str = ", ".join(f"{k}×{v}" for k, v in state.recent_categories.most_common(5))
            st.caption(f"**Categories:** {cats_str}")
        else:
            st.caption("_No session clicks yet — click items below to see live adaptation._")


def _render_recommendation(rec: Recommender, user_id: str, r: Recommendation, user_interests: set[str]) -> None:
    """Render a single Recommendation card with a click button."""
    container = st.container(border=True)
    cols = container.columns([8, 2, 1])
    with cols[0]:
        match_marker = "★ " if r.category in user_interests else ""
        container_md = (
            f"**{r.rank}.** {match_marker}{r.title}\n\n"
            f":green-background[{r.category}] {_source_badges(r.sources)}\n\n"
            f"_{r.body[:160]}{'…' if len(r.body) > 160 else ''}_"
        )
        cols[0].markdown(container_md)
    with cols[1]:
        cols[1].metric("score", f"{r.score:.3f}")
    with cols[2]:
        clicked = cols[2].button("👍", key=f"click_{user_id}_{r.item_id}", help="Click this item")
        if clicked:
            rec.on_click(user_id, r.item_id)
            st.toast(f"Clicked {r.item_id} ({r.category})", icon="✅")
            st.rerun()


# ─── page 1 — Recommendations ────────────────────────────────────────────


def page_recommendations(rec: Recommender) -> None:
    user_ids = sorted(rec.user_id_to_row.keys())
    if "current_uid" not in st.session_state:
        st.session_state.current_uid = user_ids[0]

    with st.sidebar:
        st.subheader("User")
        try:
            current_index = user_ids.index(st.session_state.current_uid)
        except ValueError:
            current_index = 0
        chosen = st.selectbox("user_id", user_ids, index=current_index, key="user_select", label_visibility="collapsed")
        st.session_state.current_uid = chosen

        c1, c2 = st.columns(2)
        if c1.button("🎲 Random", use_container_width=True):
            st.session_state.current_uid = random.choice(user_ids)
            st.rerun()
        if c2.button("↻ Reset session", use_container_width=True, help="Clear recent clicks for this user"):
            rec.user_state.reset(st.session_state.current_uid)
            rec.user_state.save()
            st.rerun()

        st.divider()
        st.subheader("Mode")
        mode = st.radio(
            "mode", ["adaptive", "long_term"], index=0,
            help="adaptive = session-blended (Phase 5); long_term = FAISS only (Phase 4)",
            label_visibility="collapsed",
        )
        k = st.slider("k", min_value=5, max_value=20, value=10, step=1)

        st.divider()
        with st.expander("➕ Sign up a new user (cold start)"):
            with st.form("signup_form", clear_on_submit=True):
                interests = st.multiselect("Interests", list(CATEGORIES))
                age_bucket = st.selectbox("Age", list(AGE_BUCKETS), index=1)
                location = st.selectbox("Location", list(LOCATIONS), index=0)
                activity = st.selectbox("Activity", list(ACTIVITY_LEVELS), index=1)
                submitted = st.form_submit_button("Sign up", use_container_width=True)
                if submitted:
                    if not interests:
                        st.error("Pick at least one interest.")
                    else:
                        new_uid = rec.add_user(
                            interests=interests, age_bucket=age_bucket,
                            location=location, activity_level=activity,
                        )
                        st.success(f"Created `{new_uid}` — switching to it.")
                        st.session_state.current_uid = new_uid
                        st.rerun()

    # main panel
    user_id = st.session_state.current_uid
    if not rec.has_user(user_id):
        st.error(f"unknown user_id: {user_id}")
        return

    _user_profile_card(rec, user_id)
    st.divider()

    profile = rec.user_profile(user_id) or {}
    user_interests = set(profile.get("interests", []))
    recs = rec.recommend(user_id, k=k, mode=mode)

    n_match = sum(1 for r in recs if r.category in user_interests)
    st.markdown(
        f"### Top {len(recs)} recommendations  ·  mode = `{mode}`  "
        f"·  {n_match}/{len(recs)} in declared interests"
    )

    for r in recs:
        _render_recommendation(rec, user_id, r, user_interests)


# ─── page 2 — Add new item ───────────────────────────────────────────────


def page_add_item(rec: Recommender) -> None:
    st.header("Add a new item (cold start)")
    st.caption(
        "Runs the trained item tower with `item_id=<UNK>` + your title/body/category, "
        "inserts into FAISS, and persists everything. The new item is recommendable immediately."
    )

    with st.form("add_item_form"):
        col1, col2 = st.columns(2)
        with col1:
            category = st.selectbox("Category", list(CATEGORIES))
            topic = st.text_input("Topic (optional)", placeholder="e.g. weekend escape")
            creator_id = st.text_input("Creator ID", value="creator_new")
        with col2:
            title = st.text_input("Title", placeholder="A weekend in Porto: cheap flights, river views")
        body = st.text_area(
            "Body",
            placeholder="Three days, walkable streets, food that ruins you for home. "
                        "60 EUR/night. Quiet shoulder season.",
            height=120,
        )
        submitted = st.form_submit_button("✨ Add item", use_container_width=True)
        if submitted:
            if not title or not body:
                st.error("Title and body are required.")
            else:
                new_iid = rec.add_item(
                    category=category, title=title, body=body,
                    topic=topic or category.lower(), creator_id=creator_id or "creator_new",
                )
                st.session_state.last_added_item = new_iid
                st.success(f"Created **`{new_iid}`** in **{category}** ✓")

    last_iid = st.session_state.get("last_added_item")
    if not last_iid or last_iid not in rec.items_by_id:
        return

    st.divider()
    new_item = rec.items_by_id[last_iid]
    new_category = new_item["category"]
    st.subheader(f"How does `{last_iid}` rank for users who like {new_category}?")

    matching_users = [
        uid for uid, u in rec.users_by_id.items()
        if new_category in u.get("interests", [])
    ][:5]

    if not matching_users:
        st.info("No users in the catalog declared this category as an interest.")
        return

    for uid in matching_users:
        rec.user_state.reset(uid)
        user_recs = rec.recommend(uid, k=20, mode="adaptive")
        match = next((r for r in user_recs if r.item_id == last_iid), None)
        profile = rec.user_profile(uid)
        interests_str = ", ".join(profile.get("interests", []))
        if match:
            st.success(
                f"`{uid}` (interests: {interests_str}) — appears at **rank {match.rank}**, "
                f"sources={match.sources}, score={match.score:.3f}"
            )
        else:
            st.warning(f"`{uid}` (interests: {interests_str}) — not in top 20")


# ─── page 3 — User state debugger ────────────────────────────────────────


def page_debugger(rec: Recommender) -> None:
    st.header("User state debugger")
    st.caption(
        "Watch the session embedding bend the recommendations. Use the "
        "Recommendations page to click items, then come back here."
    )

    user_ids = sorted(rec.user_id_to_row.keys())
    user_id = st.selectbox(
        "user_id",
        user_ids,
        index=user_ids.index(st.session_state.get("current_uid", user_ids[0]))
        if st.session_state.get("current_uid") in user_ids else 0,
        key="dbg_user_select",
    )

    profile = rec.user_profile(user_id) or {}
    state = rec.user_state.get(user_id)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Long-term profile")
        st.write(f"**Interests:** {', '.join(profile.get('interests', []))}")
        st.write(f"**Activity:** {profile.get('activity_level', '—')}")
        long_term = rec.long_term_embedding(user_id)
        st.metric("Long-term embedding norm", f"{float(np.linalg.norm(long_term)):.4f}")
        st.code("first 8 dims: " + ", ".join(f"{x:+.3f}" for x in long_term[:8]), language=None)

    with col2:
        st.subheader("Session state")
        if state and state.recent_clicked_items:
            st.write(f"**Recent clicks:** {len(state.recent_clicked_items)}")
            cats = ", ".join(f"{k}×{v}" for k, v in state.recent_categories.most_common(5))
            st.write(f"**Top categories:** {cats}")
            session_emb = rec._compute_session_embedding(state)
            if session_emb is not None:
                cos = float(np.dot(session_emb, long_term))
                st.metric("cos(long_term, session)", f"{cos:+.3f}",
                          help="1.0 = same direction, 0 = orthogonal, -1 = opposite")
                # blend weights
                n = len(state.recent_clicked_items)
                if n < 3:
                    w_long, w_sess = 0.3, 0.7
                elif n < 10:
                    w_long, w_sess = 0.5, 0.5
                else:
                    w_long, w_sess = 0.7, 0.3
                st.write(f"**Blend:** {w_long:.0%} long-term + {w_sess:.0%} session")
        else:
            st.info("No clicks recorded yet for this user. Use the Recommendations page to click some items.")

    st.divider()
    st.subheader("Side-by-side: Phase 4 (long_term) vs Phase 5 (adaptive)")

    long_term_recs = rec.recommend(user_id, k=10, mode="long_term")
    adaptive_recs = rec.recommend(user_id, k=10, mode="adaptive")
    lt_set = {r.item_id for r in long_term_recs}
    ad_set = {r.item_id for r in adaptive_recs}
    only_in_ad = ad_set - lt_set

    col_lt, col_ad = st.columns(2)
    with col_lt:
        st.markdown("**`mode=long_term`**  (FAISS only, no session blend)")
        for r in long_term_recs:
            highlight = "" if r.item_id in ad_set else " 🆕"
            st.caption(f"{r.rank:>2}. `[{r.category}]` {r.title[:55]}{highlight}")
    with col_ad:
        st.markdown("**`mode=adaptive`**  (session-blended)")
        for r in adaptive_recs:
            highlight = "" if r.item_id in lt_set else " 🆕"
            srcs = ",".join(r.sources)
            st.caption(f"{r.rank:>2}. `[{r.category}]` {r.title[:50]}  ({srcs}){highlight}")

    if only_in_ad:
        st.success(
            f"**{len(only_in_ad)} item(s)** appear ONLY in the adaptive list — "
            "that's the session blending in action. 🆕 marks them above."
        )
    else:
        st.info(
            "Both lists are identical — either no clicks have been recorded, "
            "or the session embedding aligns closely with the long-term one."
        )


# ─── page 4 — Evaluation (stub) ──────────────────────────────────────────


def page_evaluation() -> None:
    st.header("Evaluation")
    st.info(
        "Phase 8 — to be implemented. This page will render `artifacts/eval_report.json` "
        "produced by `python -m src.evaluation.evaluate`. "
        "Planned baselines: popularity, category, two-tower, two-tower + session."
    )
    st.markdown(
        "**Planned metrics**\n"
        "- Recall@10\n"
        "- Precision@10\n"
        "- MRR\n"
        "- NDCG@10\n"
    )


# ─── routing ─────────────────────────────────────────────────────────────


PAGES = {
    "🎯 Recommendations":      page_recommendations,
    "➕ Add new item":          page_add_item,
    "🔬 User state debugger":  page_debugger,
    "📊 Evaluation":           page_evaluation,
}


def main() -> None:
    st.sidebar.title("⚡ RecoForge")
    st.sidebar.caption("Real-time recommendation POC")

    ok, missing = _required_artifacts_present()
    if not ok:
        st.error("Required artifacts are missing.")
        st.code("\n".join(missing), language=None)
        st.markdown(
            "Run the pipeline first:\n\n"
            "```bash\n"
            "python -m src.data.generate_dataset\n"
            "python -m src.models.text_features\n"
            "python -m src.models.train_two_tower\n"
            "python -m src.models.export_embeddings\n"
            "python -m src.indexing.build_faiss\n"
            "```"
        )
        st.stop()

    page_label = st.sidebar.radio("Page", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()

    page_fn = PAGES[page_label]
    if page_fn is page_evaluation:
        page_fn()
    else:
        rec = get_recommender()
        page_fn(rec)


if __name__ == "__main__":
    main()
