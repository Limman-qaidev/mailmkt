# email_marketing/dashboard/mo_assistant.py
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import sqlite3
import streamlit as st

from email_marketing.analytics.recommend import get_distribution_list


_AVATAR_URL = "https://em-content.zobj.net/thumbs/240/apple/354/robot_1f916.png"


# ---------- Paths & IO ----------

def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _slugify(topic: str) -> str:
    s = "".join(ch if ch.isalnum() else "-" for ch in topic.lower().strip())
    s = "-".join([seg for seg in s.split("-") if seg])
    return s or "campaign"


def _csv_path_for_topic(topic: str) -> Path:
    return _data_dir() / f"distribution_list_{_slugify(topic)}.csv"


def _load_distribution_csv(path: Path) -> List[str]:
    """Robustly load a CSV of emails (with or w/o header)."""
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        # If there's a single column and it's not named 'email', assume it's the column
        if "email" in df.columns:
            series = df["email"]
        elif df.shape[1] == 1:
            series = df.iloc[:, 0]
        else:
            # try infer column by heuristic
            email_cols = [c for c in df.columns if "mail" in c.lower() or "email" in c.lower()]
            series = df[email_cols[0]] if email_cols else df.iloc[:, 0]
        emails = [str(v).strip() for v in series if isinstance(v, (str,)) and "@" in str(v)]
        # de-dup + order
        return sorted(dict.fromkeys(emails))
    except Exception:
        return []


def _save_distribution_csv(path: Path, emails: List[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"email": emails}).to_csv(path, index=False)
    except Exception:
        pass


def _load_all_emails() -> List[str]:
    """Fallback: distinct emails from email_map.db (ordered)."""
    db = _data_dir() / "email_map.db"
    if not db.exists():
        return []
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT recipient AS email FROM email_map ORDER BY 1"
        ).fetchall()
    emails = [r[0] for r in rows if r and r[0]]
    return sorted(dict.fromkeys(emails))


# ---------- UI helpers ----------

def _suggest_subject(topic: str) -> str:
    ts = datetime.now().strftime("%b %d, %Y")
    topic_clean = topic.strip().title() if topic.strip() else "Campaign"
    return f"{topic_clean} — Update {ts}"


def _list_stats(emails: List[str]) -> Tuple[int, List[Tuple[str, int]]]:
    size = len(emails)
    domains = [e.split("@")[-1].lower() for e in emails if "@" in e]
    top = Counter(domains).most_common(5)
    return size, top


def _prefill_and_jump(emails: List[str], subject: str, topic: str) -> None:
    st.session_state["mo_recipients"] = emails
    st.session_state["mo_subject"] = subject
    st.session_state["mo_topic"] = topic
    st.session_state["nav"] = "Email Editor"
    st.experimental_rerun()


# ---------- Page ----------

def render_mo_assistant() -> None:
    # Header / hero
    c1, c2 = st.columns([1, 5], vertical_alignment="center")
    with c1:
        st.image(_AVATAR_URL, use_container_width=True)
    with c2:
        st.title("MO Assistant")
        st.caption("Tell me the campaign topic. I’ll fetch the right audience and send you to the Email Editor.")

    # Card input
    st.markdown('<div class="mo-card">', unsafe_allow_html=True)
    topic = st.text_input(
        "Campaign topic (e.g., loans, mortgage, onboarding)",
        placeholder="loans",
        key="mo_topic_input",
    ).strip()
    col_a, col_b, col_c = st.columns([1, 1, 1])
    preview_clicked = col_a.button("Preview audience")
    proceed_clicked = col_b.button("Proceed to Email Editor", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if not (preview_clicked or proceed_clicked):
        return
    if not topic:
        st.warning("Please enter a campaign topic.")
        return

    # 1) If CSV exists in /data, use it.
    csv_path = _csv_path_for_topic(topic)
    emails: List[str] = _load_distribution_csv(csv_path)

    # 2) Else, recommendation first.
    if not emails:
        try:
            emails = list(get_distribution_list(topic, 1.0))  # threshold fijo por ahora
            emails = sorted(dict.fromkeys(e for e in emails if "@" in e))
        except Exception:
            emails = []

    # 3) Else, fallback to all emails in email_map.db
    if not emails:
        emails = _load_all_emails()

    if not emails:
        st.error("No recipients available. Please upload or generate recipients first.")
        return

    # KPIs + sample
    size, top_domains = _list_stats(emails)
    subject_default = _suggest_subject(topic)

    st.markdown('<div class="mo-card">', unsafe_allow_html=True)
    st.subheader("Audience preview")
    st.markdown(
        f"""
        <div class="mo-kpis">
          <div class="mo-kpi"><b>Total recipients</b><br>{size}</div>
          <div class="mo-kpi"><b>Topic</b><br>{topic}</div>
          <div class="mo-kpi"><b>Top domains</b><br>
            {", ".join(f"{d} ({n})" for d, n in top_domains) if top_domains else "—"}
          </div>
        </div>
        <div class="mo-muted" style="margin-top:8px">
          {'Loaded from /data CSV' if csv_path.exists() else 'Generated now'}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Show sample (first 200)"):
        st.dataframe(pd.DataFrame({"email": emails[:200]}))
    st.markdown("</div>", unsafe_allow_html=True)

    # Subject + Save
    subject = st.text_input("Suggested subject", value=subject_default, key="mo_subject_edit")
    save_csv = False
    if not csv_path.exists():  # only ask to save if it's new
        save_csv = st.checkbox(f"Save this audience to /data/{csv_path.name}", value=True)

    # Final CTA
    if st.button("Use this audience and open Email Editor", type="primary"):
        if save_csv and not csv_path.exists():
            _save_distribution_csv(csv_path, emails)
        _prefill_and_jump(emails, subject, topic)
