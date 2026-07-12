"""LLM-backed effort estimator with vector-retrieval context.

Uses pgvector nearest-neighbour tickets as few-shot grounding for the LLM.
Requires OPENAI_API_KEY; heuristic fallback is kept for tests but not used
by the Streamlit app.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from similarity_search import SimilarTicket, find_similar_tickets  # noqa: E402

FIBONACCI = [1, 2, 3, 5, 8, 13, 21]
DESCRIPTION_TRUNCATE = 300


def snap_to_fibonacci(value: float) -> int:
    """Map a value to the Fibonacci scale using the higher-bracket rule."""
    if value <= FIBONACCI[0]:
        return FIBONACCI[0]
    if value >= FIBONACCI[-1]:
        return FIBONACCI[-1]
    for low, high in zip(FIBONACCI, FIBONACCI[1:]):
        if value == low:
            return low
        if low < value < high:
            return high
    return FIBONACCI[-1]


@dataclass
class Subtask:
    title: str
    story_points: int
    reasoning: str


@dataclass
class Estimate:
    story_points: int
    confidence: float
    reasoning: str
    complex: bool = False
    subtasks: list[Subtask] = field(default_factory=list)
    similar_tickets: list[SimilarTicket] = field(default_factory=list)
    source: str = "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """You are an experienced agile delivery lead helping a team
estimate Jira tickets. For each ticket, return a JSON object with:

- story_points: an integer from the Fibonacci scale [1, 2, 3, 5, 8, 13, 21]
- confidence: a float between 0.0 and 1.0
- reasoning: 2-4 short sentences explaining the estimate, referencing
  specific aspects of the ticket (scope, unknowns, integration surface)
- complex: true if the ticket is too large for a single estimate
  (typically > 8 points or spans multiple subsystems)
- subtasks: if complex is true, an array of 2-5 subtasks, each with
  title, story_points (Fibonacci), and a one-sentence reasoning.
  Otherwise an empty array.

When similar past tickets are provided, calibrate your estimate against
their story points and explain how the current ticket compares.

Only output valid JSON. Do not invent functionality that is not implied
by the ticket description. If the description is vague, lower confidence
rather than guessing scope."""


USER_TEMPLATE = """Project: {project}
Type: {issue_type}
Title: {title}

Description:
{description}

Estimate this ticket. Respond with JSON only."""

USER_TEMPLATE_WITH_SIMILAR = """Similar past tickets (use as reference, not as copy-paste):
{similar_examples}

---
Project: {project}
Type: {issue_type}
Title: {title}

Description:
{description}

Estimate this ticket. Respond with JSON only."""


def _truncate(text: str, limit: int = DESCRIPTION_TRUNCATE) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_similar_examples(similar_tickets: list[SimilarTicket]) -> str:
    lines: list[str] = []
    for ticket in similar_tickets:
        title = ticket.title or "Untitled"
        desc = _truncate(ticket.description)
        lines.append(
            f"- {ticket.issue_key} ({ticket.story_points} SP, "
            f"similarity {ticket.similarity:.2f}): {title} — {desc}"
        )
    return "\n".join(lines)


def _build_user_prompt(ticket: dict, similar_tickets: list[SimilarTicket]) -> str:
    kwargs = {
        "project": ticket.get("project", ""),
        "issue_type": ticket.get("issue_type", ""),
        "title": ticket.get("title", ""),
        "description": ticket.get("description", ""),
    }
    if similar_tickets:
        return USER_TEMPLATE_WITH_SIMILAR.format(
            similar_examples=_format_similar_examples(similar_tickets),
            **kwargs,
        )
    return USER_TEMPLATE.format(**kwargs)


def _heuristic_estimate(ticket: dict) -> Estimate:
    """Deterministic offline estimator used when no LLM is configured."""
    text = f"{ticket.get('title','')} {ticket.get('description','')}".lower()
    length_signal = len(text.split())

    score = 0
    score += min(length_signal // 25, 6)
    big_words = ["rewrite", "migrate", "real-time", "collaborative", "redesign",
                 "architecture", "platform", "framework", "overhaul"]
    medium_words = ["implement", "add", "integrate", "support", "module",
                    "feature", "api"]
    small_words = ["fix", "typo", "rename", "update text", "log", "tweak"]

    if any(w in text for w in big_words):
        score += 6
    if any(w in text for w in medium_words):
        score += 2
    if any(w in text for w in small_words):
        score -= 2

    sp = snap_to_fibonacci(max(1, score))
    confidence = round(0.55 + min(0.25, length_signal / 400), 2)
    complex_flag = sp >= 13

    reasoning = (
        f"Heuristic signal: ~{length_signal} words in the description, "
        f"keyword profile suggests a {sp}-point scope. "
        "Configure an API key for a richer LLM-driven explanation."
    )

    subtasks: list[Subtask] = []
    if complex_flag:
        subtasks = [
            Subtask("Design and spike", 3, "Investigate approach and risks."),
            Subtask("Core implementation", 8, "Build the primary functionality."),
            Subtask("Tests and rollout", 3, "Cover edge cases and ship safely."),
        ]

    return Estimate(
        story_points=sp,
        confidence=confidence,
        reasoning=reasoning,
        complex=complex_flag,
        subtasks=subtasks,
        source="heuristic",
    )


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of an LLM response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _llm_estimate(
    ticket: dict,
    model: str,
    api_key: str,
    base_url: Optional[str],
    similar_tickets: list[SimilarTicket],
) -> Estimate:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url \
        else OpenAI(api_key=api_key)

    user = _build_user_prompt(ticket, similar_tickets)
    retrieval_suffix = "+retrieval" if similar_tickets else "+no-retrieval"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    payload = _extract_json(raw)

    sp = int(payload.get("story_points", 3))
    if sp not in FIBONACCI:
        sp = snap_to_fibonacci(float(sp))

    subtasks_raw = payload.get("subtasks") or []
    subtasks = [
        Subtask(
            title=str(s.get("title", "")).strip(),
            story_points=int(s.get("story_points", 1)),
            reasoning=str(s.get("reasoning", "")).strip(),
        )
        for s in subtasks_raw
    ]

    return Estimate(
        story_points=sp,
        confidence=float(payload.get("confidence", 0.6)),
        reasoning=str(payload.get("reasoning", "")).strip(),
        complex=bool(payload.get("complex", sp >= 13)),
        subtasks=subtasks,
        similar_tickets=similar_tickets,
        source=f"llm:{model}{retrieval_suffix}",
    )


def _fetch_similar_tickets(description: str) -> list[SimilarTicket]:
    limit = int(os.getenv("SIMILAR_TICKETS_LIMIT", "10"))
    try:
        return find_similar_tickets(description, limit=limit)
    except Exception:
        return []


def estimate_ticket(ticket: dict) -> Estimate:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("ESTIMATOR_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or None

    if not api_key:
        raise ValueError("A valid LLM API key is required")

    similar_tickets = _fetch_similar_tickets(ticket.get("description", ""))
    return _llm_estimate(ticket, model, api_key, base_url, similar_tickets)
