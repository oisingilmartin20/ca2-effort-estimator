"""Jira-styled Streamlit prototype for LLM-based effort estimation.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from estimator import Estimate, estimate_ticket
from ui.theme import setup_page

load_dotenv()

DATA_PATH = Path(__file__).parent / "data" / "tawos_sample.csv"

setup_page("backlog")


@st.cache_data
def load_tickets() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def pill_for(issue_type: str) -> str:
    cls = {
        "Story": "pill-story",
        "Bug": "pill-bug",
        "Task": "pill-task",
        "Epic": "pill-epic",
    }.get(issue_type, "pill-task")
    return f'<span class="pill {cls}">{issue_type}</span>'


def confidence_band(c: float) -> tuple[str, str]:
    if c >= 0.75:
        return "High", "#C6F09E"
    if c >= 0.55:
        return "Medium", "#F0C89E"
    return "Low", "#E8A080"


def build_comparison_df(
    estimates: dict[str, Estimate],
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for issue_key, est in estimates.items():
        match = tickets[tickets["issue_key"] == issue_key]
        if match.empty:
            continue
        ticket = match.iloc[0]
        rows.append({
            "issue_key": issue_key,
            "title": ticket["title"],
            "project": ticket["project"],
            "actual": int(ticket["actual_story_points"]),
            "estimated": int(est.story_points),
        })
    if not rows:
        return pd.DataFrame(
            columns=["issue_key", "title", "project", "actual", "estimated"],
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["project", "issue_key"])
        .reset_index(drop=True)
    )


def render_comparison_chart(df: pd.DataFrame) -> alt.Chart:
    long_df = df.melt(
        id_vars=["issue_key", "project", "title"],
        value_vars=["actual", "estimated"],
        var_name="type",
        value_name="story_points",
    )
    long_df["type"] = long_df["type"].map({
        "actual": "Actual",
        "estimated": "Estimated",
    })

    return (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("issue_key:N", title="Ticket"),
            y=alt.Y("story_points:Q", title="Story Points"),
            xOffset="type:N",
            color=alt.Color(
                "type:N",
                title="",
                scale=alt.Scale(
                    domain=["Actual", "Estimated"],
                    range=["#9EC6F0", "#A09EF0"],
                ),
            ),
            column=alt.Column("project:N", title=None),
            tooltip=["issue_key", "title", "type", "story_points"],
        )
        .properties(height=250)
    )


# --- Backlog UI -------------------------------------------------------------
tickets = load_tickets()

if "selected_key" not in st.session_state:
    st.session_state.selected_key = tickets.iloc[0]["issue_key"]
if "estimates" not in st.session_state:
    st.session_state.estimates = {}

left, right = st.columns([0.38, 0.62], gap="medium")

# --- Left: ticket list ------------------------------------------------------
with left:
    st.markdown("##### Backlog")
    project_filter = st.selectbox(
        "Project",
        ["All"] + sorted(tickets["project"].unique().tolist()),
        label_visibility="collapsed",
    )
    visible = tickets if project_filter == "All" \
        else tickets[tickets["project"] == project_filter]

    for _, row in visible.iterrows():
        is_active = row["issue_key"] == st.session_state.selected_key
        label = f"{row['issue_key']}  —  {row['title']}"
        if st.button(
            label,
            key=f"pick_{row['issue_key']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.selected_key = row["issue_key"]
            st.rerun()

# --- Right: ticket detail + estimate ---------------------------------------
with right:
    selected = tickets[tickets["issue_key"] == st.session_state.selected_key].iloc[0]

    desc_html = str(selected["description"]).replace("\n", "<br>")
    st.markdown(
        f'''
        <div class="detail-card">
          <div class="ticket-key">{selected["project"]} / {selected["issue_key"]}</div>
          <h3 style="margin:6px 0 10px 0;">{selected["title"]}</h3>
          <div style="margin-bottom:12px;">
            {pill_for(selected["issue_type"])}
            <span class="ticket-key">Team estimate:
              <b>{selected["actual_story_points"]} SP</b>
            </span>
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
        with st.spinner("Generating estimate..."):
            est = estimate_ticket({
                "project": selected["project"],
                "issue_type": selected["issue_type"],
                "title": selected["title"],
                "description": selected["description"],
            })
            st.session_state.estimates[selected["issue_key"]] = est

    est = st.session_state.estimates.get(selected["issue_key"])
    if est:
        band, color = confidence_band(est.confidence)
        actual = int(selected["actual_story_points"])
        delta = est.story_points - actual

        subtasks_html = ""
        if est.complex and est.subtasks:
            rows = "".join(
                f'<div class="subtask-card">'
                f'<b>{s.title}</b> &nbsp; '
                f'<span class="pill pill-task">{s.story_points} SP</span><br>'
                f'<span class="text-muted">{s.reasoning}</span>'
                f'</div>'
                for s in est.subtasks
            )
            subtasks_html = (
                '<div style="font-weight:600;margin-top:14px;">'
                'Suggested decomposition</div>' + rows
            )

        st.markdown(
            f'''
            <div class="estimate-card">
              <div style="display:flex;gap:24px;align-items:flex-end;
                          margin-bottom:14px;flex-wrap:wrap;">
                <div>
                  <div>Story Points</div>
                  <div class="sp-badge">{est.story_points}</div>
                </div>
                <div>
                  <div>Confidence</div>
                  <div style="color:{color};font-weight:700;font-size:18px;">
                    {band} ({est.confidence:.0%})
                  </div>
                </div>
                <div>
                  <div>Team value</div>
                  <div style="font-weight:700;font-size:18px;">
                    {actual} SP ({delta:+d})
                  </div>
                </div>
              </div>
              <div style="font-weight:600;margin-bottom:4px;">Reasoning</div>
              <div>{est.reasoning}</div>
              {subtasks_html}
              <div class="text-muted" style="margin-top:10px;font-size:12px;">
                Source: {est.source}
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    else:
        st.info("Press *Estimate Effort* to generate a recommendation.")

if st.session_state.estimates:
    st.divider()
    st.markdown("##### Estimate vs Actual by Project")
    comparison_df = build_comparison_df(st.session_state.estimates, tickets)
    st.altair_chart(
        render_comparison_chart(comparison_df),
        use_container_width=True,
    )
