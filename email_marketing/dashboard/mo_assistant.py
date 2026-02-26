# email_marketing/dashboard/mo_assistant.py
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Literal
from html import escape

import re
import pandas as pd
import sqlite3
import streamlit as st

from email_marketing.analytics.recommend import get_distribution_list


AvatarState = Literal[
    "neutral", "thinking", "speaking", "success", "warning", "error",
    "analyzing", "metrics"
]

# ------------ Session keys (preview/cache) ------------
PREVIEW_ON = "mo_preview_on"
PREVIEW_TOPIC_SLUG = "mo_preview_topic_slug"
PREVIEW_EMAILS = "mo_preview_emails"
PREVIEW_SOURCE = "mo_preview_source"  # "CSV" | "Recommender" | "Database"

# ------------ Paths & IO ------------


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _slugify(topic: str) -> str:
    s = "".join(ch if ch.isalnum() else "-" for ch in topic.lower().strip())
    s = "-".join([seg for seg in s.split("-") if seg])
    return s or "campaign"


def _csv_path_for_topic(topic: str) -> Path:
    return _data_dir() / f"distribution_list_{_slugify(topic)}.csv"


def _load_distribution_csv(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        if "email" in df.columns:
            series = df["email"]
        elif df.shape[1] == 1:
            series = df.iloc[:, 0]
        else:
            email_cols = [
                c for c in df.columns if "mail" in c.lower(
                ) or "email" in c.lower()
                ]
            series = df[email_cols[0]] if email_cols else df.iloc[:, 0]
        emails = [
            str(v).strip() for v in series if isinstance(
                v, str) and "@" in str(v)
            ]
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
    db = _data_dir() / "email_map.db"
    if not db.exists():
        return []
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT recipient AS email FROM email_map ORDER BY 1"
        ).fetchall()
    emails = [r[0] for r in rows if r and r[0]]
    return sorted(dict.fromkeys(emails))


# ------------ Avatar helpers ------------

def _load_svg(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _inline_svg(svg: str, width_px: int = 96) -> None:
    if not svg:
        return
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:center;
        width:{width_px}px;height:{width_px}px;margin:0 auto;">
          <div style="width:{width_px}px;height:{width_px}px">{svg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_mo_avatar(
        state: AvatarState = "neutral",
        width_px: int = 96
        ) -> None:
    static_dir = Path(__file__).resolve().parent / "static"
    mapping = {
        "neutral": static_dir / "mo_bot_default.svg",
        "thinking": static_dir / "mo_bot_thinking.svg",
        "speaking": static_dir / "mo_bot_speaking.svg",
        "success": static_dir / "mo_bot_success.svg",
        "warning": static_dir / "mo_bot_warning.svg",
        "error": static_dir / "mo_bot_error.svg",
        "analyzing": static_dir / "mo_bot_analyzing.svg",
        "metrics": static_dir / "mo_bot_metrics.svg",
    }
    svg_path = mapping.get(state) or mapping["neutral"]
    svg = _load_svg(svg_path)

    if not svg:
        png = static_dir / "mo.png"
        if png.exists():
            st.image(str(png), width=width_px)
            return

    if not svg:
        svg = """
        <svg width="128" height="128" viewBox="0 0 128 128"
          xmlns="http://www.w3.org/2000/svg" role="img"
            aria-label="MO assistant">
          <circle cx="64" cy="64" r="62" fill="#002147"
            stroke="#FFFFFF" stroke-width="4"/>
          <text x="64" y="78" font-size="56" font-family="Arial,
            Helvetica, sans-serif" font-weight="700"
              text-anchor="middle" fill="#FFFFFF">MO</text>
        </svg>
        """
    _inline_svg(svg, width_px=width_px)


# ------------ Recent topics (chips) ------------

def _extract_topic_from_subject(subject: str) -> str:
    if not subject:
        return ""
    s = str(subject).strip()
    m = re.match(r"^\s*\[(?P<t>[^\]]+)\]", s)
    if m:
        return m.group("t").strip()
    for sep in ("—", "-", ":"):
        if sep in s:
            return s.split(sep, 1)[0].strip()
    return " ".join(s.split()[:3]).strip()


@st.cache_data(ttl=30, show_spinner=False)
def _recent_topics_from_campaigns_db(limit: int = 6) -> List[str]:
    db = _data_dir() / "campaigns.db"
    if not db.exists():
        return []
    try:
        with sqlite3.connect(str(db)) as conn:
            cols = {row[1].lower() for row in conn.execute(
                "PRAGMA table_info(campaigns)"
                ).fetchall()}
            topics: List[str] = []
            if "topic" in cols:
                q = """
                    SELECT topic, MAX(COALESCE(created_at, updated_at)) AS ts
                    FROM campaigns
                    WHERE topic IS NOT NULL AND TRIM(topic) <> ''
                    GROUP BY topic
                    ORDER BY ts DESC
                    LIMIT ?
                """
                rows = conn.execute(q, (limit,)).fetchall()
                topics = [str(r[0]).strip() for r in rows if r and r[0]]
            elif "subject" in cols:
                q = """
                    SELECT subject, MAX(COALESCE(created_at, updated_at)) AS ts
                    FROM campaigns
                    WHERE subject IS NOT NULL AND TRIM(subject) <> ''
                    GROUP BY subject
                    ORDER BY ts DESC
                    LIMIT ?
                """
                rows = conn.execute(q, (limit * 3,)).fetchall()
                cand = [
                    _extract_topic_from_subject(
                        str(r[0])) for r in rows if r and r[0]
                    ]
                topics = [t for t in cand if t]
            seen = set()
            out: List[str] = []
            for t in topics:
                slug = _slugify(t)
                if slug not in seen:
                    seen.add(slug)
                    out.append(t)
                if len(out) >= limit:
                    break
            return out
    except Exception:
        return []


@st.cache_data(ttl=30, show_spinner=False)
def _recent_topics_from_events_db(limit: int = 6) -> List[str]:
    db = _data_dir() / "email_events.db"
    if not db.exists():
        return []
    try:
        with sqlite3.connect(str(db)) as conn:
            cols = {row[1].lower() for row in conn.execute(
                "PRAGMA table_info(events)"
                ).fetchall()}
            topics: List[str] = []
            if "subject" in cols:
                q = """
                    SELECT subject, MAX(COALESCE(timestamp, created_at)) AS ts
                    FROM events
                    WHERE subject IS NOT NULL AND TRIM(subject) <> ''
                    GROUP BY subject
                    ORDER BY ts DESC
                    LIMIT ?
                """
                rows = conn.execute(q, (limit * 3,)).fetchall()
                cand = [
                    _extract_topic_from_subject(
                        str(r[0])) for r in rows if r and r[0]
                        ]
                topics = [t for t in cand if t]
            else:
                topics = []
            seen = set()
            out: List[str] = []
            for t in topics:
                slug = _slugify(t)
                if slug not in seen:
                    seen.add(slug)
                    out.append(t)
                if len(out) >= limit:
                    break
            return out
    except Exception:
        return []


@st.cache_data(ttl=30, show_spinner=False)
def _recent_topics_from_csv(limit: int = 6) -> List[str]:
    data_dir = _data_dir()
    files = sorted(
        data_dir.glob("distribution_list_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: List[str] = []
    seen = set()
    for p in files:
        name = p.name
        topic_part = name[len("distribution_list_"): -len(".csv")]
        if not topic_part:
            continue
        label = topic_part.replace("-", " ").strip()
        slug = _slugify(label)
        if slug in seen:
            continue
        seen.add(slug)
        out.append(label)
        if len(out) >= limit:
            break
    return out


def _recent_topics(limit: int = 6) -> List[str]:
    items: List[str] = []
    items += _recent_topics_from_campaigns_db(limit=limit)
    if len(items) < limit:
        items += _recent_topics_from_events_db(limit=limit - len(items))
    if len(items) < limit:
        items += _recent_topics_from_csv(limit=limit - len(items))
    seen = set()
    out: List[str] = []
    for t in items:
        slug = _slugify(t)
        if slug not in seen:
            seen.add(slug)
            out.append(t)
        if len(out) >= limit:
            break
    return out


# ------------ UI helpers ------------

def _inject_page_css() -> None:
    st.markdown(
        """
        <style>
        .mo-hero { margin: 0 auto; max-width: 920px; }
        .mo-hero .mo-avatar { display:flex; justify-content:center; }
        .mo-hero .mo-title { text-align:center; margin-top:.35rem; }
        .mo-hero .mo-title h1 { margin: 0; }

        .mo-bubble {
            position: relative;
            background: #FFFFFF;
            border: 1px solid var(--brand);
            border-radius: 12px;
            box-shadow: var(--shadow);
            padding: .75rem 1rem;
            margin: .5rem auto 1rem auto;
            max-width: 720px;
            color: var(--text);
        }
        .mo-bubble::after {
            content: "";
            position: absolute;
            top: -10px;
            left: 50%;
            transform: translateX(-50%);
            border-width: 0 10px 10px 10px;
            border-style: solid;
            border-color: transparent transparent var(--brand) transparent;
            filter: drop-shadow(0 -1px 0 rgba(0,0,0,0.05));
        }
        .mo-bubble::before {
            content: "";
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            border-width: 0 9px 9px 9px;
            border-style: solid;
            border-color: transparent transparent #FFFFFF transparent;
        }

        .mo-card { background: var(--card); border-radius: var(--radius);
          box-shadow: var(--shadow); padding: 1rem; }

        .mo-chip-row { display:flex; flex-wrap:wrap; gap:.5rem;
          margin:.25rem 0 .75rem 0; }
        .mo-chip button {
            border-radius: 999px !important;
            background: #FFFFFF !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            padding: .25rem .75rem !important;
        }
        .mo-chip button:hover {
            border-color: var(--brand) !important;
            color: var(--brand) !important;
        }

        /* Stat tiles */
        .mo-stat {
            background:#FFFFFF; border:1px solid var(--border);
            border-radius: 12px; box-shadow: var(--shadow);
            padding: .9rem 1rem; display:flex; gap:.75rem; align-items:center;
            min-height: 86px;
        }
        .mo-stat .icon {
            width: 40px; height: 40px; border-radius: 999px;
            display:flex; align-items:center; justify-content:center;
            background: #EEF2FF; border:1px solid #DBEAFE;
            font-size: 1.1rem;
        }
        .mo-stat .meta { line-height: 1.1; }
        .mo-stat .label { font-size:.75rem; color: var(--muted);
          letter-spacing:.04em; text-transform:uppercase; }
        .mo-stat .value { font-size:1.5rem; font-weight:700;
          color: var(--text); margin-top:.2rem; }

        .mo-badge {
            display:inline-block; padding: .2rem .6rem; border-radius: 999px;
            background: #EEF2FF; color: #0F172A; border: 1px solid #DBEAFE;
            font-size: .75rem; vertical-align: middle;
        }
        .mo-domain {
            display:inline-block; padding:.15rem .5rem; border-radius:999px;
            background:#F3F4F6; border:1px solid #E5E7EB;
              margin:.15rem .25rem 0 0;
            font-size:.78rem;
        }
        .mo-section-head { display:flex; align-items:center; gap:.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _suggest_subject(topic: str) -> str:
    ts = datetime.now().strftime("%b %d, %Y")
    topic_clean = topic.strip().title() if topic.strip() else "Campaign"
    return f"{topic_clean} — Update {ts}"


def _list_stats(emails: List[str]) -> Tuple[int, List[Tuple[str, int]]]:
    size = len(emails)
    domains = [e.split("@")[-1].lower() for e in emails if "@" in e]
    top = Counter(domains).most_common(5)
    return size, top


# -------- Preview state helpers --------

def _clear_preview() -> None:
    st.session_state.pop(PREVIEW_ON, None)
    st.session_state.pop(PREVIEW_TOPIC_SLUG, None)
    st.session_state.pop(PREVIEW_EMAILS, None)
    st.session_state.pop(PREVIEW_SOURCE, None)


def _set_preview(topic: str, emails: List[str], source: str) -> None:
    st.session_state[PREVIEW_ON] = True
    st.session_state[PREVIEW_TOPIC_SLUG] = _slugify(topic)
    st.session_state[PREVIEW_EMAILS] = emails
    st.session_state[PREVIEW_SOURCE] = source


def _prefill_and_jump(
        emails: List[str],
        subject_to_use: str,
        topic: str
        ) -> None:
    st.session_state["mo_recipients"] = emails
    st.session_state["mo_subject"] = (subject_to_use or "").strip()
    st.session_state["mo_topic"] = topic
    st.session_state["nav_redirect"] = "Email Editor"
    st.rerun()


def _set_avatar_state(state: AvatarState) -> None:
    st.session_state["mo_state"] = state


def _get_avatar_state(default: AvatarState = "neutral") -> AvatarState:
    return st.session_state.get("mo_state", default)


def _render_audience_preview(
        topic: str, emails: List[str], source_label: str) -> None:
    """Centered, product-like preview tiles with source badge and sample.
    CSS is injected on every render to survive page switches."""
    # 1) ALWAYS inject CSS (no session flag); switching pages drops previous
    #  CSS.
    st.markdown(
        """
        <style>
          .mo-wrap      { max-width: 980px; margin: 0 auto; }
          .mo-head      { display:flex; align-items:center;
            gap:.5rem; margin:.2rem 0 1rem 0; }
          .mo-badge     { display:inline-block; padding:.18rem .6rem;
            border-radius:999px;
                           background:#EEF2FF; color:#0F172A;
                             border:1px solid #DBEAFE; font-size:.75rem; }
          .mo-tiles     { display:grid; grid-template-columns: repeat(3, 1fr);
            gap:16px; }
          .mo-tile      { background:#FFFFFF; border:1px solid var(--border);
            border-radius:12px;
                           box-shadow:var(--shadow); padding:16px;
                             text-align:center; min-height:100px; }
          .mo-ico       { width:44px; height:44px; border-radius:999px;
            display:inline-flex;
                           align-items:center; justify-content:center;
                             background:#EEF2FF;
                           border:1px solid #DBEAFE; font-size:22px;
                             margin-bottom:6px; }
          .mo-label     { font-size:.74rem; color:var(--muted);
            text-transform:uppercase; letter-spacing:.04em; }
          .mo-value     { font-size:1.55rem; font-weight:700;
            color:var(--text); margin-top:.2rem; }
          .mo-domains   { margin-top:.35rem; }
          .mo-domain    { display:inline-block; padding:.15rem .5rem;
            border-radius:999px;
                           background:#F3F4F6; border:1px solid #E5E7EB;
                             margin:.15rem .25rem 0 0; font-size:.78rem; }
          @media (max-width: 900px){ .mo-tiles { grid-template-columns: 1fr;
            } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2) Stats
    size, top_domains = _list_stats(emails)
    top3 = top_domains[:3]
    extra = max(0, len(top_domains) - 3)

    # 3) Header with source badge (centered container)
    st.markdown('<div class="mo-wrap">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mo-head"><h3 style="margin:0">Audience preview</h3>'
        f'<span class="mo-badge">Source: {escape(source_label or "—")}'
        '</span></div>',
        unsafe_allow_html=True,
    )

    # 4) Three tiles
    topic_safe = escape(topic or "—")
    domains_html = " ".join(
        f'<span class="mo-domain">{escape(d)} ({n})</span>' for d, n in top3
        ) or "—"
    if extra > 0:
        domains_html += f' <span class="mo-domain">+{extra} more</span>'

    st.markdown(
        f"""
        <div class="mo-tiles">
          <div class="mo-tile">
            <div class="mo-ico">👥</div>
            <div class="mo-label">Recipients</div>
            <div class="mo-value">{size:,}</div>
          </div>

          <div class="mo-tile">
            <div class="mo-ico">🏷️</div>
            <div class="mo-label">Topic</div>
            <div class="mo-value">{topic_safe}</div>
          </div>

          <div class="mo-tile">
            <div class="mo-ico">🌐</div>
            <div class="mo-label">Top domains</div>
            <div class="mo-value" style="font-size:1.05rem; font-weight:600;">
            &nbsp;</div>
            <div class="mo-domains">{domains_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5) Optional sample (kept centered by wrapper)
    with st.expander("Show sample (first 200)"):
        st.dataframe(pd.DataFrame({"email": emails[:200]}))

    st.markdown("</div>", unsafe_allow_html=True)  # close .mo-wrap

# ========================== Page ==========================


def render_mo_assistant() -> None:
    """Render MO landing with deterministic preview tiles on every click."""
    _inject_page_css()

    # ---------- Rehydrate seed BEFORE creating the input ----------
    seed = st.session_state.pop("mo_topic_seed", None)
    if seed:
        st.session_state["mo_topic_input"] = seed
    elif not st.session_state.get("mo_topic_input"):
        payload = st.session_state.get("mo_preview_payload")
        if payload and isinstance(payload, dict):
            st.session_state["mo_topic_input"] = payload.get("topic_label", "")

    # ---------- Avatar heuristic ----------
    avatar_state: AvatarState = "thinking"
    if st.session_state.get("mo_recipients"):
        avatar_state = "success"
    avatar_state = _get_avatar_state(avatar_state)

    # ---------- HERO ----------
    st.markdown('<div class="mo-hero">', unsafe_allow_html=True)
    st.markdown('<div class="mo-avatar">', unsafe_allow_html=True)
    _render_mo_avatar(state=avatar_state, width_px=128)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="mo-title">', unsafe_allow_html=True)
    st.header("Hi, I’m MO")
    st.markdown("</div>", unsafe_allow_html=True)
    msg = {
        "neutral": "What would you like to do?",
        "thinking": "Tell me the campaign topic. I’ll prepare your audience.",
        "speaking": "Working on it… preparing your audience.",
        "success": "All set. I can preload recipients and subject for you.",
        "warning": "I need a valid topic or a larger audience to proceed.",
        "error": "Something went wrong building the audience. Try again.",
    }[avatar_state]
    st.markdown(f'<div class="mo-bubble">{msg}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- INPUT CARD ----------
    st.markdown('<div class="mo-card">', unsafe_allow_html=True)

    # Chips (seed + rerun seguro)
    recent = _recent_topics(limit=6)
    if recent:
        st.caption("Recent topics")
        st.markdown('<div class="mo-chip-row">', unsafe_allow_html=True)
        cols = st.columns(min(6, len(recent)))
        for i, label in enumerate(recent):
            col = cols[i % len(cols)]
            friendly = label.strip()
            slug = _slugify(friendly)
            with col:
                if st.button(friendly.title(), key=f"chip_{slug}"):
                    st.session_state["mo_topic_seed"] = friendly
                    st.session_state["mo_subject_live"] = _suggest_subject(
                        friendly)
                    st.session_state.pop("mo_preview_payload", None)
                    if hasattr(st, "toast"):
                        st.toast(f"Topic loaded: {friendly}")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    topic_input = st.text_input(
        "Campaign topic",
        placeholder="e.g., loans, mortgages, onboarding",
        key="mo_topic_input",
    ).strip()
    topic_label = topic_input or (
        st.session_state.get("mo_preview_payload") or {}
        ).get("topic_label", "")
    topic_slug = _slugify(topic_label)

    # Si el topic cambia respecto al cache, invalídalo
    cached = st.session_state.get("mo_preview_payload")
    if (
        cached and isinstance(
            cached, dict) and cached.get("topic_slug") != topic_slug
    ):
        st.session_state.pop("mo_preview_payload", None)
        cached = None

    col_a, col_b, _ = st.columns([1, 1, 1])
    preview_clicked = col_a.button("Preview audience")
    proceed_clicked = col_b.button("Proceed to Email Editor", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)  # close input card

    # Helper local: construye audiencia y devuelve (emails, source_label)
    def _build_audience(_topic: str) -> tuple[List[str], str]:
        if not _topic:
            return ([], "")
        _set_avatar_state("speaking")
        csv_path = _csv_path_for_topic(_topic)
        with st.spinner("Preparing audience..."):
            # 1) CSV
            emails = _load_distribution_csv(csv_path)
            if emails:
                return (emails, "CSV")
            # 2) Recommender
            try:
                emails = list(get_distribution_list(_topic, 1.0))
                emails = sorted(dict.fromkeys(e for e in emails if isinstance(
                    e, str) and "@" in e))
            except Exception:
                emails = []
            if emails:
                return (emails, "Recommender")
            # 3) DB fallback
            emails = _load_all_emails()
            if emails:
                return (emails, "Database")
        return ([], "")

    # ---------- Decide qué mostrar/construir ----------
    emails: List[str] = []
    source_label: str = ""
    show_preview: bool = False

    if preview_clicked:
        if not topic_label:
            _set_avatar_state("warning")
            st.warning("Please enter a campaign topic.")
            return
        emails, source_label = _build_audience(topic_label)
        if not emails:
            _set_avatar_state("warning")
            st.error(
                "No recipients available. Please upload or generate recipients"
                " first."
                )
            return
        # Guarda payload y muestra SIEMPRE en esta ejecución
        payload = {
            "topic_label": topic_label,
            "topic_slug": topic_slug,
            "emails": emails,
            "source": source_label,
        }
        st.session_state["mo_preview_payload"] = payload
        show_preview = True

    elif (
        cached and isinstance(
            cached, dict) and cached.get("topic_slug") == topic_slug
    ):
        # Reutiliza cache SOLO si corresponde al topic actual
        emails = list(cached.get("emails", []) or [])
        source_label = str(cached.get("source", "") or "")
        show_preview = bool(emails)

    # ---------- Navegación directa ----------
    subject_default = _suggest_subject(topic_label)
    subject_live = st.session_state.get("mo_subject_live", subject_default)

    if proceed_clicked:
        # Si no hay emails cargados para este topic, constrúyelos ahora
        if not emails:
            emails, source_label = _build_audience(topic_label)
            if not emails:
                _set_avatar_state("warning")
                st.error(
                    "No recipients available. Please upload or generate"
                    " recipients first."
                    )
                return
            st.session_state["mo_preview_payload"] = {
                "topic_label": topic_label,
                "topic_slug": topic_slug,
                "emails": emails,
                "source": source_label,
            }
        _prefill_and_jump(emails, subject_live, topic_label)
        return

    # ---------- PREVIEW VISUAL (solo si hay algo que mostrar) ----------
    if show_preview:
        _render_audience_preview(topic_label, emails, source_label)

        # Subject vivo
        subject_live = st.text_input(
            "Suggested subject",
            value=st.session_state.get("mo_subject_live", subject_default),
            key="mo_subject_edit",
            help="You can edit this and it will be prefilled in the Email"
            " Editor.",
        )
        st.session_state["mo_subject_live"] = subject_live

        # Guardado opcional si no existe CSV
        csv_path = _csv_path_for_topic(topic_label)
        ask_save = (not csv_path.exists()) and (source_label != "CSV")
        save_csv = st.checkbox(f"Save this audience to /data/{csv_path.name}",
                               value=True) if ask_save else False

        if (
            st.button("Use this audience and open Email Editor",
                      type="primary")
        ):
            if save_csv and (not csv_path.exists()):
                _save_distribution_csv(csv_path, emails)
            _prefill_and_jump(emails, subject_live, topic_label)
    else:
        # No preview que mostrar aún (espera a que el usuario pulse Preview)
        return
