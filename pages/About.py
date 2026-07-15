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
        Create tickets on the <b>Ticket</b> tab, then switch to <b>Backlog</b> to
        select one and generate estimates (story points, confidence, reasoning, and
        optional subtask breakdown).
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
        <li><b>Backlog</b> — user-created tickets stored in
            <code>data/user_tickets.json</code></li>
        <li><b>Estimation</b> — <code>estimator.py</code> computes story points from
            pgvector neighbours (description similarity), then calls an OpenAI-compatible
            chat API for reasoning and optional decomposition</li>
        <li><b>Retrieval</b> — pgvector similarity search over the TAWOS embedding
            corpus; when no neighbours are found, the LLM estimates alone with
            low confidence and a clear warning</li>
        <li><b>Fallback</b> — deterministic keyword/heuristic estimator in
            <code>estimator.py</code> for offline tests (not used by the Streamlit app)</li>
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
        <li>TAWOS data is used for <b>vector retrieval context</b> during estimation
            and for offline dataset analytics — it is not loaded as the app backlog.</li>
        <li>At inference time, only the <b>selected ticket's title and
            description</b> are sent to the LLM API. This tool does not perform
            batch training or model weight updates.</li>
        <li>User tickets are persisted in <code>data/user_tickets.json</code>.
            Session estimates are stored in <b>browser session state only</b>.</li>
        <li>If no API key is configured, the heuristic fallback runs entirely
            <b>offline</b> with no external data transfer.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)
