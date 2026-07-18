"""Label RQ4 hallucination review rows with local Olmo (same model as estimator).

Uses the grounded / inferred / hallucinated rubric from rq4_prepare_review.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

LABELS = {"grounded", "inferred", "hallucinated"}

SYSTEM = """You label agile ticket subtasks for hallucination risk.
Reply with JSON only: {"label": "grounded"|"inferred"|"hallucinated", "why": "one short sentence"}

Definitions:
- grounded: the subtask is explicitly supported by the ticket title or description
- inferred: a reasonable decomposition step implied by the ticket, not invented scope
- hallucinated: invents systems, features, stakeholders, or work the ticket never suggests
"""

USER = """Ticket title: {title}

Ticket description:
{description}

Subtask title: {subtask_title}
Subtask reasoning: {subtask_reasoning}

Classify the subtask."""


def _extract_label(raw: str) -> str:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON")
    text = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    payload = json.loads(text)
    label = str(payload.get("label", "")).strip().lower()
    if label not in LABELS:
        raise ValueError(f"bad label {label!r}")
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("evaluation/hallucination_review_rag.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/hallucination_review_rag_labeled.csv"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    model = os.getenv("ESTIMATOR_MODEL", "olmo-3-7b-instruct")
    if "api.groq.com" in (base_url or "").lower():
        print("Refusing Groq endpoint.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.review)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    client = OpenAI(api_key=api_key, base_url=base_url)
    labels: list[str] = []

    for i, row in df.iterrows():
        user = USER.format(
            title=row["ticket_title"],
            description=str(row["ticket_description"])[:4000],
            subtask_title=row["subtask_title"],
            subtask_reasoning=row["subtask_reasoning"],
        )
        label = None
        last_err = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                )
                label = _extract_label(resp.choices[0].message.content or "")
                break
            except Exception as exc:
                last_err = exc
                time.sleep(0.5)
        if label is None:
            print(f"[{i + 1}/{len(df)}] FAIL {row['issue_key']}: {last_err}", flush=True)
            label = "inferred"  # conservative fallback if parse fails
        labels.append(label)
        print(f"[{i + 1}/{len(df)}] {row['issue_key']} -> {label}", flush=True)

    df = df.copy()
    df["label"] = labels
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} labeled rows to {args.out}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
