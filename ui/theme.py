"""Shared Streamlit theme, header, and navigation."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

import streamlit as st

PageId = Literal["backlog", "about"]

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.jpg"


def _logo_data_uri() -> str:
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def inject_theme() -> None:
    st.markdown(
        """
        <style>
          :root {
            --color-1: #9EC6F0;
            --color-2: #A09EF0;
            --color-2-dark: #8886D9;
            --color-3: #C89EF0;
            --color-4: #F0C89E;
            --color-5: #C6F09E;
            --jira-bg: #FAFBFC;
            --jira-card: #FFFFFF;
            --jira-border: #E8EAF0;
            --jira-text: #172B4D;
            --jira-muted: #5E6C84;
          }
          #MainMenu, footer, [data-testid="stToolbar"], .stAppDeployButton {
            visibility: hidden;
            display: none;
          }
          section[data-testid="stSidebar"] {
            display: none;
          }
          header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
          }
          .stApp { background: var(--jira-bg); color: var(--jira-text); }
          .block-container { padding-top: 1rem; color: var(--jira-text); }
          .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
          .stApp h5, .stApp h6, .stApp li, .stApp label, .stApp span,
          .stApp div[data-testid="stMarkdownContainer"],
          .stApp div[data-testid="stMarkdownContainer"] * {
            color: var(--jira-text);
          }
          div[data-testid="stMarkdownContainer"] h5 {
            color: var(--color-3) !important;
          }
          .detail-card, .detail-card * { color: var(--jira-text) !important; }
          .estimate-card, .estimate-card * { color: var(--jira-text); }
          .subtask-card, .subtask-card b { color: var(--jira-text); }
          .text-muted { color: var(--jira-muted); }
          .stSelectbox label, .stSelectbox div { color: var(--jira-text); }
          div[data-baseweb="select"] *, div[data-baseweb="select"] input {
            color: var(--jira-text) !important;
          }
          div[data-baseweb="popover"] li { color: var(--jira-text) !important; }
          .jira-header, .jira-header * {
            color: var(--jira-text) !important;
          }
          .jira-header {
            background: var(--color-2);
            height: 5.5rem;
            min-height: 5.5rem;
            padding: 0 28px 0 0;
            border-radius: 8px;
            margin-bottom: 12px;
            font-weight: 600;
            font-size: 1.6rem;
            line-height: 1.3;
            display: flex;
            align-items: center;
            gap: 16px;
            overflow: hidden;
          }
          .jira-header .logo-img {
            height: 100%;
            width: auto;
            display: block;
            object-fit: contain;
            border-radius: 0;
            background: white;
            padding: 0;
            margin: 0;
          }
          .jira-header .title {
            font-weight: 600;
            font-size: 1.6rem;
          }
          .nav-row {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            width: 100%;
          }
          .nav-row .nav-btn {
            flex: 1 1 0;
            min-width: 0;
            display: block;
            box-sizing: border-box;
            text-align: center;
            padding: 12px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 1rem;
            background: var(--jira-card) !important;
            color: var(--jira-text) !important;
            border: 1px solid var(--jira-border) !important;
            text-decoration: none !important;
            cursor: pointer;
          }
          .nav-row a.nav-btn:link,
          .nav-row a.nav-btn:visited,
          .nav-row a.nav-btn:active {
            color: var(--jira-text) !important;
            text-decoration: none !important;
            background: var(--jira-card) !important;
            border: 1px solid var(--jira-border) !important;
          }
          .nav-row .nav-btn.nav-active,
          .nav-row span.nav-btn.nav-active {
            background: var(--color-2) !important;
            border-color: var(--color-2) !important;
            color: var(--jira-text) !important;
            text-decoration: none !important;
          }
          .nav-row a.nav-btn:hover {
            background: var(--color-1) !important;
            border-color: var(--color-2) !important;
            color: var(--jira-text) !important;
            text-decoration: none !important;
          }
          div[data-testid="stMarkdownContainer"] .nav-row a,
          div[data-testid="stMarkdownContainer"] .nav-row span {
            color: var(--jira-text) !important;
            text-decoration: none !important;
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
            border-left: 3px solid var(--color-2);
            background: var(--color-1);
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
          .pill-story { background: var(--color-5); color: var(--jira-text); }
          .pill-bug   { background: var(--color-4); color: var(--jira-text); }
          .pill-task  { background: var(--color-1); color: var(--jira-text); }
          .pill-epic  { background: var(--color-3); color: var(--jira-text); }
          .detail-card {
            background: var(--jira-card);
            border: 1px solid var(--jira-border);
            border-radius: 6px;
            padding: 18px 22px;
          }
          .estimate-card {
            background: #FFFFFF;
            border: 1px solid var(--jira-border);
            border-left: 4px solid var(--color-2);
            padding: 18px 22px;
            border-radius: 4px;
            margin-top: 24px;
          }
          .detail-card { margin-bottom: 18px; }
          div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }
          .sp-badge {
            background: var(--color-2);
            color: var(--jira-text);
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
            background: var(--color-2);
            color: var(--jira-text);
            border: none;
            font-weight: 600;
          }
          .stButton button:hover {
            background: var(--color-2-dark);
            color: var(--jira-text);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_page(active: PageId) -> None:
    """Configure layout and render shared chrome. Must be the first Streamlit call."""
    st.set_page_config(
        page_title="Effort Estimator",
        page_icon=str(LOGO_PATH),
        layout="wide",
    )
    inject_theme()
    render_page_header()
    render_nav(active)


def render_page_header() -> None:
    logo = _logo_data_uri()
    st.markdown(
        f'<div class="jira-header">'
        f'<img class="logo-img" src="{logo}" alt="Effort Estimator logo">'
        f'<span class="title">Effort Estimator</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_nav(active: PageId) -> None:
    if active == "backlog":
        backlog = '<span class="nav-btn nav-active">Backlog</span>'
        about = '<a class="nav-btn" href="/About" target="_self">About</a>'
    else:
        backlog = '<a class="nav-btn" href="/" target="_self">Backlog</a>'
        about = '<span class="nav-btn nav-active">About</span>'
    st.markdown(
        f'<div class="nav-row">{backlog}{about}</div>',
        unsafe_allow_html=True,
    )
