"""Jira-styled Streamlit prototype for LLM-based effort estimation.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from estimator import estimate_ticket
from tickets.store import load_tickets
from ui.components import (
    confidence_band,
    escape_html,
    pill_for,
    render_rag_baseline_html,
    render_similar_tickets_html,
)
from ui.theme import setup_page

load_dotenv()

setup_page("backlog")

if "selected_key" not in st.session_state:
    st.session_state.selected_key = None
if "estimates" not in st.session_state:
    st.session_state.estimates = {}

tickets = load_tickets()

st.markdown("##### Backlog")

if tickets.empty:
    st.info("No tickets yet. Create one on the **Ticket** tab, then return here to estimate it.")
else:
    key_to_title = dict(zip(tickets["issue_key"], tickets["title"]))
    keys = tickets["issue_key"].tolist()

    if st.session_state.selected_key not in keys:
        st.session_state.selected_key = keys[0]

    selected_key = st.selectbox(
        "Backlog",
        options=keys,
        index=keys.index(st.session_state.selected_key),
        format_func=lambda k: f"{k} — {key_to_title[k]}",
        label_visibility="collapsed",
    )
    st.session_state.selected_key = selected_key

    selected = tickets[tickets["issue_key"] == selected_key].iloc[0]
    desc_html = escape_html(selected["description"])

    st.markdown(
        f'''
        <div class="detail-card">
          <div class="ticket-key">{escape_html(selected["project"])} / {escape_html(selected["issue_key"])}</div>
          <h3 style="margin:6px 0 10px 0;">{escape_html(selected["title"])}</h3>
          <div style="margin-bottom:12px;">
            {pill_for(selected["issue_type"])}
          </div>
          <div style="font-weight:600;margin-bottom:4px;">Description</div>
          <div>{desc_html}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    btn_col, _ = st.columns([0.3, 0.7])
    with btn_col:
        run = st.button("Estimate Effort", use_container_width=True)

    if run:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("A valid LLM API key is required. Set OPENAI_API_KEY in .env.")
        elif not os.getenv("ESTIMATOR_MODEL"):
            st.error("Set ESTIMATOR_MODEL in .env (e.g. olmo-3-7b-instruct).")
        else:
            with st.spinner("Generating estimate..."):
                try:
                    est = estimate_ticket({
                        "project": selected["project"],
                        "issue_type": selected["issue_type"],
                        "title": selected["title"],
                        "description": selected["description"],
                    })
                    st.session_state.estimates[selected["issue_key"]] = est
                except Exception as exc:
                    st.error(f"Estimation failed: {exc}")

    est = st.session_state.estimates.get(selected["issue_key"])
    if est:
        if est.no_rag_fallback:
            st.warning(
                "No similar tickets were found in the vector store. "
                "Story points were estimated from ticket details only, "
                "with low confidence."
            )

        band, color = confidence_band(est.confidence)
        rag_baseline_html = render_rag_baseline_html(est)

        subtasks_html = ""
        if est.complex and est.subtasks:
            rows = "".join(
                f'<div class="subtask-card">'
                f'<b>{escape_html(s.title)}</b> &nbsp; '
                f'<span class="pill pill-task">{s.story_points} SP</span><br>'
                f'<span class="text-muted">{escape_html(s.reasoning)}</span>'
                f'</div>'
                for s in est.subtasks
            )
            subtasks_html = (
                '<div style="font-weight:600;margin-top:14px;">'
                'Suggested decomposition</div>' + rows
            )

        similar_html = render_similar_tickets_html(est)

        st.markdown(
            f'''
            <div class="estimate-card">
              <div style="display:flex;gap:24px;align-items:flex-end;
                          margin-bottom:14px;flex-wrap:wrap;">
                <div>
                  <div>Story Points</div>
                  <div class="sp-badge">{est.story_points}</div>
                  {rag_baseline_html}
                </div>
                <div>
                  <div>Confidence</div>
                  <div style="color:{color};font-weight:700;font-size:18px;">
                    {band} ({est.confidence:.0%})
                  </div>
                </div>
              </div>
              {similar_html}
              <div style="font-weight:600;margin-bottom:4px;">Reasoning</div>
              <div>{escape_html(est.reasoning)}</div>
              {subtasks_html}
              <div class="text-muted" style="margin-top:10px;font-size:12px;">
                Source: {escape_html(est.source)}
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        if est.no_rag_fallback:
            st.info(
                "Run generate_embeddings.py and ensure Postgres is running "
                "to enable RAG-based estimates."
            )
    else:
        st.info("Press *Estimate Effort* to generate a recommendation.")
