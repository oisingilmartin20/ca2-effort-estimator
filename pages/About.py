"""About page — aim, technology stack, and data statement."""
from __future__ import annotations

import streamlit as st

from ui.theme import setup_page

setup_page("about")

st.markdown(
    """
    <div class="detail-card">
      <h3 style="margin-top:0;">Aim of this tool</h3>
      <p>
        This CA2 prototype supports <b>LLM-assisted agile story point estimation</b>.
        It lets you browse a Jira-style backlog, generate estimates (story points,
        confidence, reasoning, and optional subtask breakdown), and compare AI
        estimates against <b>team-recorded actual story points</b> using the
        comparison chart on the Backlog page.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="detail-card">
      <h3 style="margin-top:0;">What this tool uses</h3>
      <ul>
        <li><b>UI</b> — Streamlit with a custom pastel CSS theme</li>
        <li><b>Data</b> — <code>data/tawos_sample.csv</code> (14 sample tickets
            across 5 open-source projects from the TAWOS dataset)</li>
        <li><b>Estimation</b> — <code>estimator.py</code> calls an OpenAI-compatible
            chat API when <code>OPENAI_API_KEY</code> is set (default model in
            <code>.env.example</code>: <code>llama-3.3-70b-versatile</code> via Groq)</li>
        <li><b>Fallback</b> — deterministic keyword/heuristic estimator when no
            API key is configured or the LLM call fails</li>
        <li><b>Charts</b> — Altair grouped bar chart (estimated vs actual, faceted
            by project)</li>
        <li><b>Scale</b> — Fibonacci story points: 1, 2, 3, 5, 8, 13, 21</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="detail-card">
      <h3 style="margin-top:0;">Data and training statement</h3>
      <ul>
        <li>The model is <b>not fine-tuned</b> on TAWOS or any project data in
            this repository.</li>
        <li>TAWOS sample tickets are used as a <b>demonstration and evaluation
            dataset</b> only — they provide real historical team estimates
            (<code>actual_story_points</code>) for comparison.</li>
        <li>At inference time, only the <b>selected ticket's title and
            description</b> are sent to the LLM API. This tool does not perform
            batch training or model weight updates.</li>
        <li>Session estimates are stored in <b>browser session state only</b> and
            are not written to disk.</li>
        <li>If no API key is configured, the heuristic fallback runs entirely
            <b>offline</b> with no external data transfer.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
