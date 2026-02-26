"""Email editor component for the Streamlit dashboard.

This module defines a Streamlit view that lets users upload a recipient
list, compose an HTML message, and trigger the sending of a campaign via
SMTP (or Mailgun if later enabled). Recipient lists can be provided in CSV
or Excel format and are displayed back to the user for verification.

MO integration:
- If MO Assistant preloaded a list/subject, a third source mode
  "From MO Assistant (preloaded)" becomes available automatically.
- During sending, session flags are set so the sidebar avatar switches to
  the "writing" animation (mo_bot_writing.svg).
"""

from __future__ import annotations

import os
import time
import urllib.parse
import uuid
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import streamlit as st
import textwrap

from email_marketing.ab_testing import assign_variant
from email_marketing.analytics.recommend import get_distribution_list
# from email_marketing.mailer.mailgun_sender import MailgunSender
from email_marketing.mailer.smtp_sender import SMTPSender


# ============================ Utilities ============================

def _now_ts() -> str:
    """UTC timestamp as 'YYYY-mm-dd HH:MM:SS.ffffff' (no 'T')."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")


def _load_recipients(upload: Optional[Any]) -> List[str]:
    """Load recipient addresses from an uploaded CSV/Excel (first column).

    Non-email values are ignored. Returns a list of strings.
    """
    if upload is None:
        return []
    try:
        if upload.name.lower().endswith(".csv"):
            df = pd.read_csv(upload)
        else:
            df = pd.read_excel(upload)
    except Exception as exc:
        st.error(f"Failed to parse file: {exc}")
        return []

    emails: List[str] = []
    for val in df.iloc[:, 0].astype(str):
        if "@" in val:
            emails.append(val.strip())
    return emails


def _db_paths_for_send() -> tuple[str, str]:
    """Return paths to email_events.db and email_map.db under /data."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    events_db = str(data_dir / "email_events.db")
    email_map_db = str(data_dir / "email_map.db")
    return events_db, email_map_db


def _upsert_email_map(
    email_map_db: str,
    msg_id: str,
    recipient: str,
    variant: str | None,
    ts_str: str,
) -> None:
    """Insert/replace row in email_map with backward-compatible schema."""
    with sqlite3.connect(email_map_db) as conn:
        cols = [r[1].lower() for r in conn.execute("PRAGMA table_info(email_map)").fetchall()]
        has_send_ts = "send_ts" in cols

        if has_send_ts:
            conn.execute(
                "INSERT OR REPLACE INTO email_map (msg_id, recipient, variant, send_ts) "
                "VALUES (?, ?, ?, ?)",
                (msg_id, recipient, variant, ts_str),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO email_map (msg_id, recipient, variant) "
                "VALUES (?, ?, ?)",
                (msg_id, recipient, variant),
            )
        conn.commit()


def _log_send_event(
    events_db: str,
    msg_id: str,
    campaign: str,
    client_ip: str = "0.0.0.0",
    ts_str: str | None = None,
) -> None:
    """Insert a 'send' event into events table (schema-compatible)."""
    if ts_str is None:
        ts_str = _now_ts()
    with sqlite3.connect(events_db) as conn:
        conn.execute(
            """
            INSERT INTO events (msg_id, event_type, client_ip, ts, campaign)
            VALUES (?, 'send', ?, ?, ?)
            """,
            (msg_id, client_ip, ts_str, campaign),
        )
        conn.commit()


# ================== MO: sending flags for sidebar avatar ==================

def _mo_set_sending_flags(value: bool) -> None:
    """Set/clear sending flags so the sidebar avatar switches to 'writing'."""
    for key in ("email_sending", "campaign_sending", "sending"):
        st.session_state[key] = bool(value)


@contextmanager
def mo_sending_state() -> None:
    """Context manager to toggle sending flags during the send window."""
    _mo_set_sending_flags(True)
    try:
        yield
    finally:
        _mo_set_sending_flags(False)


# ============================ Main view ============================
def _normalize_email(email: str) -> str:
    """Lowercase/trim and validate a minimal email shape."""
    s = str(email).strip().lower()
    if "@" in s and "." in s.split("@")[-1]:
        return s
    return ""


def _render_recipient_manager(base_emails: List[str]) -> List[str]:
    """Search and exclude recipients interactively. Returns the final list to send."""
    st.subheader("Manage recipients")

    # Normalize base list (deduplicate)
    base_norm = []
    seen = set()
    for e in base_emails:
        ne = _normalize_email(e)
        if ne and ne not in seen:
            seen.add(ne)
            base_norm.append(ne)

    # Session exclusions (persist across reruns/pages)
    excl_key = "recipient_exclusions"
    exclusions: set[str] = set(st.session_state.get(excl_key, []))

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        query = st.text_input(
            "Search/filter recipients",
            placeholder="e.g. jane@, @example.com, 'mortgage'",
            help="Type a substring or @domain to filter the preview below.",
            key="recip_search",
        ).strip().lower()

    with col2:
        st.metric("Total in list", f"{len(base_norm):,}")
    with col3:
        st.metric("Excluded", f"{len(exclusions):,}")

    # Filter for preview
    if query:
        if query.startswith("@"):
            dom = query[1:]
            filtered = [e for e in base_norm if e.endswith("@" + dom) or e.split("@")[-1] == dom]
        else:
            filtered = [e for e in base_norm if query in e]
    else:
        filtered = base_norm

    # Multiselect (capped for performance)
    cap = 1500
    options = filtered[:cap]
    selected = st.multiselect(
        f"Select emails to exclude (showing up to {cap:,} matches)",
        options=options,
        default=[],
        key="recip_multiexclude",
    )

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        if st.button("Add to exclusions"):
            exclusions.update(_normalize_email(e) for e in selected)
            st.session_state[excl_key] = sorted(exclusions)
            st.success(f"Added {len(selected)} to exclusions.")
    with cB:
        if st.button("Clear exclusions", help="Remove all manual exclusions."):
            exclusions.clear()
            st.session_state[excl_key] = []
            st.info("Exclusions cleared.")

    # Manual exclude box
    manual = st.text_area(
        "Manual exclusions (comma/space/newline separated)",
        placeholder="paste or type emails here…",
        height=80,
        key="recip_manual_excl",
    )
    if st.button("Exclude listed emails"):
        added = 0
        for raw in re.split(r"[,\s]+", manual):
            ne = _normalize_email(raw)
            if ne:
                if ne not in exclusions:
                    added += 1
                exclusions.add(ne)
        st.session_state[excl_key] = sorted(exclusions)
        st.success(f"Added {added} email(s) to exclusions.")

    # Preview table (filtered view) and exclusions summary
    with st.expander("Preview (filtered)"):
        st.dataframe(pd.DataFrame({"email": options}))
    if exclusions:
        chips = ", ".join(list(sorted(exclusions))[:10])
        more = max(0, len(exclusions) - 10)
        st.caption(f"Excluded (first 10): {chips}" + (f"  ·  +{more} more" if more else ""))

    # Final list to send = base - exclusions
    final_emails = [e for e in base_norm if e not in exclusions]
    st.success(f"Final recipients to send: {len(final_emails):,}")
    return final_emails


def render_email_editor() -> None:
    """Render the email editor page (stable across reruns) with recipient manager."""
    import re  # for manual exclusions parsing

    st.header("Email Campaign Editor")

    # ---------- Persistent session state ----------
    if "recipient_base" not in st.session_state:
        st.session_state["recipient_base"] = []        # working list (persistent)
    if "recipient_exclusions" not in st.session_state:
        st.session_state["recipient_exclusions"] = []  # exclusions (persistent)
    if "preview_list" not in st.session_state:
        st.session_state["preview_list"] = []          # preview from recommender
    if "editor_subject" not in st.session_state:
        st.session_state["editor_subject"] = ""        # sticky subject

    # ---------- Prefill from MO (consume once) ----------
    incoming_mo = st.session_state.pop("mo_recipients", None)
    mo_subject_live = st.session_state.pop(
        "mo_subject_live", st.session_state.pop("mo_subject", "")
    )
    mo_topic = st.session_state.pop("mo_topic", "")

    # If MO provided recipients in this run, set them as the working base
    if incoming_mo:
        base = [str(e).strip().lower() for e in incoming_mo if "@" in str(e)]
        base = [e for e in base if "." in e.split("@")[-1]]  # minimal validation
        # deduplicate preserving order
        dedup: list[str] = []
        seen = set()
        for e in base:
            if e not in seen:
                seen.add(e)
                dedup.append(e)
        st.session_state["recipient_base"] = dedup
        if mo_subject_live and not st.session_state["editor_subject"]:
            st.session_state["editor_subject"] = mo_subject_live

    # ---------- Modes (load/replace the working base) ----------
    has_mo = bool(incoming_mo or st.session_state.get("recipient_base"))
    modes = ["Upload list"]
    if has_mo:
        modes.insert(0, "From MO Assistant (preloaded)")
    mode = st.radio("Recipient source", modes, index=0 if has_mo else 0)

    def _normalize_email(email: str) -> str:
        s = str(email).strip().lower()
        return s if ("@" in s and "." in s.split("@")[-1]) else ""

    def _show_preview_table(emails: list[str], label: str) -> None:
        st.caption(label)
        st.dataframe(pd.DataFrame({"email": emails}))

    # ---- Mode: From MO (base ya cargada si venía esta ejecución) ----
    if mode == "From MO Assistant (preloaded)":
        base_now = st.session_state["recipient_base"]
        st.success(
            f"MO preloaded {len(base_now)} recipients"
            f"{(' for topic: ' + mo_topic) if mo_topic else ''}."
        )
        if base_now:
            doms = sorted({e.split("@")[-1] for e in base_now})
            dom_sel = st.multiselect(
                "Filter by domain (visual preview only)", doms, key="mo_domain_filter"
            )
            view = [e for e in base_now if not dom_sel or e.split("@")[-1] in dom_sel]
            _show_preview_table(view, "Preview of current working list")
        st.markdown("---")

    # ---- Mode: Upload list (reemplaza base cuando hay fichero) ----
    elif mode == "Upload list":
        upload = st.file_uploader(
            "Upload recipient list (CSV or Excel)",
            type=["csv", "xls", "xlsx"],
        )
        if upload is not None:
            up = _load_recipients(upload)
            up_norm = []
            seen = set()
            for e in up:
                ne = _normalize_email(e)
                if ne and ne not in seen:
                    seen.add(ne)
                    up_norm.append(ne)
            st.session_state["recipient_base"] = up_norm
            st.success(f"Loaded {len(up_norm)} recipients into the working list.")
            _show_preview_table(up_norm, "Preview of current working list")
        st.markdown("---")

    # ---- Mode: Recommender (preview → Apply to working base) ----
    else:
        campaign_id = st.text_input(
            "Campaign ID",
            help="Identifier of the campaign used to build the recommended list.",
        )
        colp, cola = st.columns([1, 1])
        preview_clicked = colp.button("Preview")
        apply_clicked = cola.button("Use preview as working list")

        if preview_clicked:
            if not campaign_id:
                st.warning("Please provide a Campaign ID.")
            else:
                try:
                    recs = list(get_distribution_list(campaign_id, 1.0))
                    rec_norm = []
                    seen = set()
                    for e in recs:
                        ne = _normalize_email(e)
                        if ne and ne not in seen:
                            seen.add(ne)
                            rec_norm.append(ne)
                    st.session_state["preview_list"] = rec_norm
                    st.success(f"Loaded {len(rec_norm)} recommended recipients (preview).")
                except Exception as exc:
                    st.error(f"Recommendation failed: {exc}")

        if st.session_state["preview_list"]:
            doms = sorted({e.split("@")[-1] for e in st.session_state["preview_list"]})
            dom_sel = st.multiselect(
                "Filter preview by domain",
                doms,
                key="rec_domain_filter",
            )
            filtered = [
                e
                for e in st.session_state["preview_list"]
                if not dom_sel or e.split("@")[-1] in dom_sel
            ]
            _show_preview_table(filtered, "Preview (recommender)")
            if apply_clicked:
                st.session_state["recipient_base"] = filtered
                st.success(f"Applied {len(filtered)} recipients to the working list.")
        st.markdown("---")

    # ================== Manage recipients (tabs; non-destructive) ==================
    base_norm: list[str] = list(st.session_state["recipient_base"])
    if not base_norm:
        st.info("Load or generate a recipient list to manage and send.")
        return

    st.subheader("Manage recipients")

    exclusions: set[str] = set(st.session_state.get("recipient_exclusions", []))

    # Métricas superiores
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption("Manage who will receive this campaign.")
    with col2:
        st.metric("Total in working list", f"{len(base_norm):,}")
    with col3:
        st.metric("Excluded", f"{len(exclusions):,}")

    # Tabs: selección individual y (opcional) exclusión manual
    tab_select, tab_manual = st.tabs(["Select individually", "Manual exclusions"])

    with tab_select:
        query = st.text_input(
            "Search/filter (visual)",
            placeholder="e.g. jane@, @example.com, 'mortgage'",
            help="Type a substring or @domain to narrow the list below, then select and exclude.",
            key="recip_search",
        ).strip().lower()

        # Vista filtrada (no altera la base)
        if query:
            if query.startswith("@"):
                dom = query[1:]
                filtered = [
                    e for e in base_norm
                    if e.endswith("@" + dom) or e.split("@")[-1] == dom
                ]
            else:
                filtered = [e for e in base_norm if query in e]
        else:
            filtered = base_norm

        cap = 2000
        options = filtered[:cap]
        selected = st.multiselect(
            f"Select emails to exclude (showing up to {cap:,} matches)",
            options=options,
            default=[],
            key="recip_multiexclude",
        )

        cA, cB, cC = st.columns([1, 1, 1])
        with cA:
            if st.button("Add selected to exclusions"):
                added = 0
                for e in selected:
                    if e not in exclusions:
                        exclusions.add(e)
                        added += 1
                st.session_state["recipient_exclusions"] = sorted(exclusions)
                st.success(f"Added {added} email(s) to exclusions.")
        with cB:
            if query.startswith("@") and st.button("Exclude this domain"):
                dom = query[1:]
                added = 0
                for e in base_norm:
                    if e.endswith("@" + dom) or e.split("@")[-1] == dom:
                        if e not in exclusions:
                            exclusions.add(e)
                            added += 1
                st.session_state["recipient_exclusions"] = sorted(exclusions)
                st.success(f"Excluded domain @{dom} ({added} addresses).")
        with cC:
            if st.button("Clear exclusions"):
                exclusions.clear()
                st.session_state["recipient_exclusions"] = []
                st.info("Exclusions cleared.")

    with tab_manual:
        manual = st.text_area(
            "Manual exclusions (comma/space/newline separated)",
            placeholder="paste or type emails here…",
            height=100,
            key="recip_manual_excl",
        )
        if st.button("Exclude listed emails"):
            added = 0
            for raw in re.split(r"[,\s]+", manual):
                ne = raw.strip().lower()
                if "@" in ne and "." in ne.split("@")[-1]:
                    if ne not in exclusions:
                        exclusions.add(ne)
                        added += 1
            st.session_state["recipient_exclusions"] = sorted(exclusions)
            st.success(f"Added {added} email(s) to exclusions.")

    # ---------- Final list (base − exclusions) ----------
    final_recipients: list[str] = [e for e in base_norm if e not in exclusions]
    st.success(f"Final recipients to send: {len(final_recipients):,}")

    # Side-by-side preview: Final vs Excluded
    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("Final recipients (after exclusions)", expanded=False):
            # Cap visual para rendimiento
            df_final = pd.DataFrame({"email": final_recipients[:3000]})
            st.dataframe(df_final, use_container_width=True)
    with col_right:
        with st.expander("Excluded recipients (audit)", expanded=False):
            excl_sorted = sorted(exclusions)
            df_excl = pd.DataFrame({"email": excl_sorted[:3000]})
            st.dataframe(df_excl, use_container_width=True)

            # NEW: select excluded emails to re-include (undo exclusion)
            to_include = st.multiselect(
                "Select emails to re-include",
                options=excl_sorted[:3000],
                default=[],
                key="recip_multiinclude",
                help="Pick excluded addresses to add back to the final recipients."
            )
            c_inc1, c_inc2 = st.columns([1, 1])
            with c_inc1:
                if st.button("Re-include selected", key="btn_reinclude"):
                    # Remove chosen emails from the exclusions set
                    before = len(exclusions)
                    exclusions.difference_update(to_include)
                    st.session_state["recipient_exclusions"] = sorted(exclusions)
                    st.success(f"Re-included {before - len(exclusions)} address(es).")
            with c_inc2:
                if st.button("Clear exclusions", key="btn_clear_all_excl"):
                    exclusions.clear()
                    st.session_state["recipient_exclusions"] = []
                    st.info("Exclusions cleared.")


    st.markdown("---")

    # ================== Compose & Send ==================
    default_subject = st.session_state["editor_subject"] or mo_subject_live or ""
    subject_value = st.text_input(
        "Subject", max_chars=200, value=default_subject, key="subject_input"
    )
    st.session_state["editor_subject"] = subject_value  # sticky

    html_body = st.text_area(
        "HTML Body",
        height=300,
        placeholder="<p>Hello {{ name }}, welcome to our newsletter.</p>",
        key="html_body",
    )

    can_send = bool(final_recipients) and bool(subject_value) and bool(html_body)
    send_button = st.button(
        "Send Email", type="primary", disabled=not can_send, key="mo_send_button"
    )

    if not send_button:
        return

    # ---------- Send ----------
    sender = SMTPSender()
    tracking_url = os.environ.get(
        "TRACKING_URL", "https://track.jonathansalgadonieto.com"
    ).strip()

    total = len(final_recipients)
    progress = st.progress(0.0)
    events_db_path, email_map_db_path = _db_paths_for_send()

    # Optional: MO animation during send, if available
    try:
        ctx = mo_sending_state()  # type: ignore[name-defined]
    except Exception:
        ctx = None
    if ctx:
        cm = ctx
    else:
        from contextlib import nullcontext
        cm = nullcontext()

    with cm:
        with st.spinner("Sending emails..."):
            for i, email in enumerate(final_recipients, start=1):
                variant = assign_variant(email)
                msg_id = uuid.uuid4().hex
                timestamp = int(time.time())

                # Query strings
                logo_qs = urllib.parse.urlencode(
                    {"msg_id": msg_id, "ts": timestamp, "campaign": subject_value}
                )
                click_qs = urllib.parse.urlencode(
                    {"msg_id": msg_id, "url": "https://example.com", "campaign": subject_value}
                )
                unsub_qs = urllib.parse.urlencode({"msg_id": msg_id, "campaign": subject_value})
                comp_qs = urllib.parse.urlencode({"msg_id": msg_id, "campaign": subject_value})

                links_row = (
                    f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin-top:8px;">'
                    f'<tr>'
                    f'<td style="padding-right:16px;"><a href="{tracking_url}/click?{click_qs}">Click here</a></td>'
                    f'<td style="padding-right:16px;"><a href="{tracking_url}/unsubscribe?{unsub_qs}">Unsubscribe</a></td>'
                    f'<td><a href="{tracking_url}/complaint?{comp_qs}">Report spam</a></td>'
                    f'</tr>'
                    f'</table>'
                )

                full_html = textwrap.dedent(f"""\
                <!DOCTYPE html>
                <html>
                <head><meta charset="utf-8"></head>
                <body>
                <div>{html_body}</div>
                <div style="margin:12px 0 4px 0;">
                <img src="{tracking_url}/logo?{logo_qs}" alt="Company Logo" width="200" style="display:block;"/>
                </div>
                {links_row}
                </body>
                </html>""")

                try:
                    sender.send_email(
                        recipient=email,
                        msg_id=msg_id,
                        html=full_html,
                        subject=subject_value,
                        variant=variant,
                    )
                    ts_str = _now_ts()
                    _log_send_event(events_db_path, msg_id, subject_value, "0.0.0.0", ts_str)
                    _upsert_email_map(email_map_db_path, msg_id, email, variant, ts_str)
                except Exception as exc:
                    st.error(f"Failed to send to {email}: {exc}")

                progress.progress(i / total)

    if hasattr(st, "toast"):
        st.toast(f"Campaign sent to {total} recipients.")
    st.success("Campaign sent.")
