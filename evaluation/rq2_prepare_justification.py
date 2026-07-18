"""RQ2 justification step 1 - build a sheet to label RAG reasoning by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RUBRIC_GUIDE = (
    "# Score each rubric column 0 or 1:\n"
    "#   grounded          - claims match the ticket text and neighbours\n"
    "#   faithful_to_rag   - explains why those neighbours support the SP\n"
    "#   comparative       - says how the current ticket is similar/different\n"
    "#   no_hallucination  - no invented features/systems\n"
    "#   useful            - a human estimator would find it actionable\n"
)

RUBRIC_COLUMNS = [
    "grounded",
    "faithful_to_rag",
    "comparative",
    "no_hallucination",
    "useful",
]


def _parse_neighbours(raw) -> list[dict]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def build_review_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        neighbours = _parse_neighbours(row.get("similar_tickets_json"))
        if not neighbours:
            continue

        neighbour_keys = ", ".join(
            f"{t.get('issue_key')} ({float(t.get('similarity', 0)):.2f})"
            for t in neighbours
        )
        description = str(row.get("description", "") or "")
        snippet = description[:500] + ("..." if len(description) > 500 else "")

        entry = {
            "issue_key": row["issue_key"],
            "ticket_title": row.get("title", ""),
            "ticket_description_snippet": snippet,
            "predicted_story_points": row.get("predicted_story_points", ""),
            "reasoning": row.get("reasoning", ""),
            "neighbour_keys": neighbour_keys,
            "similar_tickets_json": row.get("similar_tickets_json", "[]"),
        }
        for col in RUBRIC_COLUMNS:
            entry[col] = ""
        rows.append(entry)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("evaluation/results_rag.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/justification_review.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    review_df = build_review_rows(df)

    if review_df.empty:
        print("No RAG rows with neighbours in this results file — nothing to review.")
        return

    review_df.to_csv(args.out, index=False)
    print(RUBRIC_GUIDE)
    print(f"Wrote {len(review_df)} justification rows to {args.out}")
    print("Fill in the rubric columns (0/1), then run rq2_justification.py.")


if __name__ == "__main__":
    main()
