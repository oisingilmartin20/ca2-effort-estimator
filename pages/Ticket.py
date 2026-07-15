"""Ticket creation page — add new tickets to the backlog."""
from __future__ import annotations

import streamlit as st

from tickets.store import add_ticket
from ui.constants import ISSUE_TYPES
from ui.theme import setup_page

setup_page("ticket")

st.markdown("##### New Ticket")

with st.form("new_ticket", clear_on_submit=True):
    title = st.text_input("Title")
    project = st.text_input("Project Name")
    issue_type = st.selectbox("Task Type", ISSUE_TYPES)
    description = st.text_area("Description", height=200)
    submitted = st.form_submit_button("Add to Backlog", use_container_width=True)

if submitted:
    if not title.strip() or not project.strip() or not description.strip():
        st.error("Title, Project Name, and Description are required.")
    else:
        issue_key = add_ticket(project, issue_type, title, description)
        st.success(f"Added **{issue_key}** to the backlog. Switch to the Backlog tab to estimate it.")
