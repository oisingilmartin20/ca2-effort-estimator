"""RAG-first effort estimator with LLM explanation.

Story points are computed deterministically from pgvector neighbours when
available; the LLM provides reasoning, confidence, and optional decomposition.
Requires OPENAI_API_KEY for the Streamlit app.
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
NO_RAG_CONFIDENCE_CAP = 0.45
NO_RAG_WARNING = (
    "No similar tickets found in the vector store; "
    "estimate is based on ticket details only."
)

SYSTEM_PROMPT_EXPLAIN = """You are an experienced agile delivery lead helping a team
estimate Jira tickets. The story points have already been computed from similar
past tickets. Return a JSON object with:

- confidence: a float between 0.0 and 1.0 (lower if the description is vague)
- reasoning: 2-4 short sentences explaining the estimate. Cite the top 1-2
  similar past tickets by issue key and explain how their story points informed
  the fixed estimate and how the current ticket compares in scope.
- complex: true if the ticket is too large for a single estimate
  (typically > 8 points or spans multiple subsystems)
- subtasks: if complex is true, an array of 2-5 subtasks, each with
  title, story_points (Fibonacci), and a one-sentence reasoning.
  Subtask story_points must sum to the provided RAG total.
  Otherwise an empty array.

Do NOT return story_points — it is fixed. Only output valid JSON."""

SYSTEM_PROMPT_NO_RAG = """You are an experienced agile delivery lead helping a team
estimate Jira tickets. No similar past tickets were found in the vector store.
Return a JSON object with:

- story_points: an integer from the Fibonacci scale [1, 2, 3, 5, 8, 13, 21]
- confidence: a float between 0.0 and 1.0 (keep low — no retrieval context)
- reasoning: 2-4 short sentences explaining the estimate from ticket scope alone
- complex: true if the ticket is too large for a single estimate
- subtasks: if complex is true, an array of 2-5 subtasks with Fibonacci points.
  Otherwise an empty array.

Only output valid JSON. Do not invent functionality not implied by the ticket."""

USER_TEMPLATE_EXPLAIN = """RAG-computed story points: {rag_story_points}
(weighted average {rag_raw_average:.1f} from {n_similar} similar tickets)

Similar past tickets:
{similar_examples}

---
Project: {project}
Type: {issue_type}
Title: {title}

Description:
{description}

Explain this estimate. Respond with JSON only."""

USER_TEMPLATE_NO_RAG = """Project: {project}
Type: {issue_type}
Title: {title}

Description:
{description}

Estimate this ticket without retrieval context. Respond with JSON only."""


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


def compute_rag_story_points(
    similar_tickets: list[SimilarTicket],
) -> tuple[int, float, float]:
    """Return (fibonacci_sp, raw_weighted_avg, rag_confidence)."""
    min_sim = float(os.getenv("RAG_MIN_SIMILARITY", "0"))
    tickets = [t for t in similar_tickets if t.similarity >= min_sim]
    if not tickets:
        tickets = similar_tickets

    weight_sum = sum(t.similarity for t in tickets)
    if weight_sum <= 0:
        raw = float(tickets[0].story_points)
    else:
        raw = sum(t.story_points * t.similarity for t in tickets) / weight_sum

    sp = snap_to_fibonacci(raw)
    top_sim = max(t.similarity for t in tickets)
    sp_values = [t.story_points for t in tickets]
    sp_spread = max(sp_values) - min(sp_values)

    sim_factor = min(top_sim / 0.85, 1.0)
    spread_penalty = min(sp_spread / 13.0, 0.5)
    rag_conf = max(0.3, min(0.9, 0.3 + 0.6 * sim_factor - spread_penalty))

    return sp, raw, round(rag_conf, 2)


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
    rag_story_points: Optional[int] = None
    rag_raw_average: Optional[float] = None
    rag_confidence: Optional[float] = None
    no_rag_fallback: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


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


def _ticket_kwargs(ticket: dict) -> dict[str, str]:
    return {
        "project": ticket.get("project", ""),
        "issue_type": ticket.get("issue_type", ""),
        "title": ticket.get("title", ""),
        "description": ticket.get("description", ""),
    }


def _extract_json(raw: str) -> dict:
    """Pull the first JSON object out of an LLM response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _parse_subtasks(payload: dict) -> list[Subtask]:
    subtasks_raw = payload.get("subtasks") or []
    return [
        Subtask(
            title=str(s.get("title", "")).strip(),
            story_points=int(s.get("story_points", 1)),
            reasoning=str(s.get("reasoning", "")).strip(),
        )
        for s in subtasks_raw
    ]


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url \
        else OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    return _extract_json(raw)


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


def _llm_explain_rag_estimate(
    ticket: dict,
    model: str,
    api_key: str,
    base_url: Optional[str],
    similar_tickets: list[SimilarTicket],
    rag_sp: int,
    raw_avg: float,
    rag_conf: float,
) -> Estimate:
    user = USER_TEMPLATE_EXPLAIN.format(
        rag_story_points=rag_sp,
        rag_raw_average=raw_avg,
        n_similar=len(similar_tickets),
        similar_examples=_format_similar_examples(similar_tickets),
        **_ticket_kwargs(ticket),
    )
    payload = _call_llm(SYSTEM_PROMPT_EXPLAIN, user, model, api_key, base_url)

    llm_conf = float(payload.get("confidence", rag_conf))
    confidence = min(llm_conf, rag_conf)
    reasoning = str(payload.get("reasoning", "")).strip()
    subtasks = _parse_subtasks(payload)

    return Estimate(
        story_points=rag_sp,
        confidence=confidence,
        reasoning=reasoning,
        complex=bool(payload.get("complex", rag_sp >= 13)),
        subtasks=subtasks,
        similar_tickets=similar_tickets,
        source=f"rag+llm:{model}",
        rag_story_points=rag_sp,
        rag_raw_average=round(raw_avg, 2),
        rag_confidence=rag_conf,
        no_rag_fallback=False,
    )


def _llm_estimate_without_rag(
    ticket: dict,
    model: str,
    api_key: str,
    base_url: Optional[str],
) -> Estimate:
    user = USER_TEMPLATE_NO_RAG.format(**_ticket_kwargs(ticket))
    payload = _call_llm(SYSTEM_PROMPT_NO_RAG, user, model, api_key, base_url)

    sp = int(payload.get("story_points", 3))
    if sp not in FIBONACCI:
        sp = snap_to_fibonacci(float(sp))

    confidence = min(float(payload.get("confidence", 0.4)), NO_RAG_CONFIDENCE_CAP)
    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning.startswith(NO_RAG_WARNING):
        reasoning = f"{NO_RAG_WARNING} {reasoning}"

    return Estimate(
        story_points=sp,
        confidence=confidence,
        reasoning=reasoning,
        complex=bool(payload.get("complex", sp >= 13)),
        subtasks=_parse_subtasks(payload),
        similar_tickets=[],
        source=f"llm:{model}+no-retrieval",
        no_rag_fallback=True,
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

    if similar_tickets:
        rag_sp, raw_avg, rag_conf = compute_rag_story_points(similar_tickets)
        return _llm_explain_rag_estimate(
            ticket, model, api_key, base_url,
            similar_tickets, rag_sp, raw_avg, rag_conf,
        )

    return _llm_estimate_without_rag(ticket, model, api_key, base_url)
