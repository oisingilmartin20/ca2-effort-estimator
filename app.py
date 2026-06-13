"""Jira-styled Streamlit prototype for LLM-based effort estimation.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from estimator import estimate_ticket

load_dotenv()

DATA_PATH = Path(__file__).parent / "data" / "tawos_sample.csv"

st.set_page_config(
    page_title="Effort Estimator",
    page_icon=":bar_chart:",
    layout="wide",
)

# --- Jira-style theming -----------------------------------------------------
st.markdown(
    """
    <style>
      :root {
        --jira-blue: #0052CC;
        --jira-blue-dark: #0747A6;
        --jira-bg: #F4F5F7;
        --jira-card: #FFFFFF;
        --jira-border: #DFE1E6;
        --jira-text: #172B4D;
        --jira-muted: #5E6C84;
      }
      .stApp { background: var(--jira-bg); color: var(--jira-text); }
      .block-container { padding-top: 3.5rem; color: var(--jira-text); }
      header[data-testid="stHeader"] { background: transparent; }
      .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
      .stApp h5, .stApp h6, .stApp li, .stApp label, .stApp span,
      .stApp div[data-testid="stMarkdownContainer"],
      .stApp div[data-testid="stMarkdownContainer"] * {
        color: var(--jira-text);
      }
      .detail-card, .detail-card * { color: var(--jira-text) !important; }
      .estimate-card, .estimate-card * { color: var(--jira-text); }
      .subtask-card, .subtask-card b { color: var(--jira-text); }
      .stSelectbox label, .stSelectbox div { color: var(--jira-text); }
      div[data-baseweb="select"] *, div[data-baseweb="select"] input {
        color: var(--jira-text) !important;
      }
      div[data-baseweb="popover"] li { color: var(--jira-text) !important; }
      .jira-header {
        background: var(--jira-blue);
        color: white;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .jira-header .logo {
        background: white;
        color: var(--jira-blue);
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 13px;
      }
      .ticket-row {
        background: var(--jira-card);
        border: 1px solid var(--jira-border);
        border-radius: 4px;
        padding: 10px 12px;
        margin-bottom: 6px;
        cursor: pointer;
      }
      .ticket-row.active {
        border-left: 3px solid var(--jira-blue);
        background: #DEEBFF;
      }
      .ticket-key {
        font-family: 'Consolas', monospace;
        font-size: 12px;
        color: var(--jira-muted);
      }
      .ticket-title { color: var(--jira-text); font-weight: 500; }
      .pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 6px;
      }
      .pill-story { background: #E3FCEF; color: #006644; }
      .pill-bug   { background: #FFEBE6; color: #BF2600; }
      .pill-task  { background: #DEEBFF; color: #0747A6; }
      .pill-epic  { background: #EAE6FF; color: #403294; }
      .detail-card {
        background: var(--jira-card);
        border: 1px solid var(--jira-border);
        border-radius: 6px;
        padding: 18px 22px;
      }
      .estimate-card {
        background: #FFFFFF;
        border: 1px solid var(--jira-border);
        border-left: 4px solid var(--jira-blue);
        padding: 18px 22px;
        border-radius: 4px;
        margin-top: 24px;
      }
      .detail-card { margin-bottom: 18px; }
      div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }
      .sp-badge {
        background: var(--jira-blue);
        color: white;
        padding: 6px 14px;
        border-radius: 18px;
        font-weight: 700;
        font-size: 18px;
      }
      .subtask-card {
        background: white;
        border: 1px solid var(--jira-border);
        border-radius: 4px;
        padding: 10px 12px;
        margin-top: 6px;
      }
      .stButton button {
        background: var(--jira-blue);
        color: white;
        border: none;
        font-weight: 600;
      }
      .stButton button:hover { background: var(--jira-blue-dark); color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        return "High", "#36B37E"
    if c >= 0.55:
        return "Medium", "#FFAB00"
    return "Low", "#DE350B"


# --- Header -----------------------------------------------------------------
st.markdown(
    '<div class="jira-header">'
    '<span class="logo">EE</span>'
    'Effort Estimator &nbsp;/&nbsp; TAWOS Backlog'
    '</div>',
    unsafe_allow_html=True,
)

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
                f'<span style="color:#5E6C84;">{s.reasoning}</span>'
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
              <div style="margin-top:10px;font-size:12px;color:#5E6C84;">
                Source: {est.source}
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    else:
        st.info("Press *Estimate Effort* to generate a recommendation.")
