"""LLM-backed effort estimator with a deterministic offline fallback.

The fallback lets the prototype run without an API key (useful for grading
demos and development), while the live path uses an OpenAI-compatible
chat completions endpoint.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

FIBONACCI = [1, 2, 3, 5, 8, 13, 21]


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
    source: str = "heuristic"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


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

Only output valid JSON. Do not invent functionality that is not implied
by the ticket description. If the description is vague, lower confidence
rather than guessing scope."""


USER_TEMPLATE = """Project: {project}
Type: {issue_type}
Title: {title}

Description:
{description}

Estimate this ticket. Respond with JSON only."""


def _heuristic_estimate(ticket: dict) -> Estimate:
    """Deterministic offline estimator used when no LLM is configured.

    Uses a simple length + keyword signal so the demo is usable end-to-end
    without network access.
    """
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

    sp = min(FIBONACCI, key=lambda f: abs(f - max(1, score)))
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


def _llm_estimate(ticket: dict, model: str, api_key: str,
                  base_url: Optional[str]) -> Estimate:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url \
        else OpenAI(api_key=api_key)

    user = USER_TEMPLATE.format(
        project=ticket.get("project", ""),
        issue_type=ticket.get("issue_type", ""),
        title=ticket.get("title", ""),
        description=ticket.get("description", ""),
    )

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
        sp = min(FIBONACCI, key=lambda f: abs(f - sp))

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
        source=f"llm:{model}",
    )


def estimate_ticket(ticket: dict) -> Estimate:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("ESTIMATOR_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or None

    if not api_key:
        return _heuristic_estimate(ticket)

    try:
        return _llm_estimate(ticket, model, api_key, base_url)
    except Exception as exc:
        fallback = _heuristic_estimate(ticket)
        fallback.reasoning = (
            f"LLM call failed ({exc.__class__.__name__}); showing heuristic "
            f"estimate. {fallback.reasoning}"
        )
        return fallback
