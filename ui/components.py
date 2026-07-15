"""Shared Streamlit UI helpers."""
from __future__ import annotations

from estimator import Estimate


def pill_for(issue_type: str) -> str:
    cls = {
        "Story": "pill-story",
        "Bug": "pill-bug",
        "Task": "pill-task",
        "Epic": "pill-epic",
        "Improvement": "pill-improvement",
        "New Feature": "pill-new-feature",
        "Suggestion": "pill-suggestion",
        "Enhancement Request": "pill-enhancement",
        "Technical task": "pill-technical",
    }.get(issue_type, "pill-task")
    return f'<span class="pill {cls}">{issue_type}</span>'


def confidence_band(c: float) -> tuple[str, str]:
    if c >= 0.75:
        return "High", "#C6F09E"
    if c >= 0.55:
        return "Medium", "#F0C89E"
    return "Low", "#E8A080"


def escape_html(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def render_rag_baseline_html(est: Estimate) -> str:
    if est.rag_story_points is None or est.rag_raw_average is None:
        return ""
    n = len(est.similar_tickets)
    return (
        '<div class="text-muted" style="font-size:12px;margin-top:4px;">'
        f"Based on {n} similar ticket{'s' if n != 1 else ''} "
        f"(weighted avg {est.rag_raw_average:.1f} → {est.rag_story_points} SP)"
        "</div>"
    )


def render_similar_tickets_html(est: Estimate) -> str:
    if not est.similar_tickets:
        return ""

    rows = []
    for ticket in est.similar_tickets:
        title = escape_html(ticket.title or "Untitled")
        desc = escape_html(ticket.description[:300])
        if len(ticket.description) > 300:
            desc += "..."
        rows.append(
            f'<div class="subtask-card">'
            f'<b>{escape_html(ticket.issue_key)}</b> &nbsp; '
            f'<span class="pill pill-task">{ticket.story_points} SP</span> &nbsp; '
            f'<span class="text-muted">({ticket.similarity:.0%} similar)</span><br>'
            f'<span style="font-weight:600;">{title}</span><br>'
            f'<span class="text-muted">{desc}</span>'
            f"</div>"
        )

    return (
        '<div style="font-weight:600;margin-top:14px;margin-bottom:8px;">'
        "Similar past tickets</div>"
        + "".join(rows)
    )
