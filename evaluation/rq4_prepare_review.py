"""RQ4 step 1 - turn generated subtasks into a sheet we can label by hand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LABEL_GUIDE = (
    "# label each row:\n"
    "#   grounded     - actually in the ticket text\n"
    "#   inferred     - reasonable guess, not explicit but not made up either\n"
    "#   hallucinated - scope the ticket never mentioned\n"
)


def expand_subtasks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        subtasks = json.loads(row["subtasks_json"]) if row["subtasks_json"] else []
        for subtask in subtasks:
            rows.append({
                "issue_key": row["issue_key"],
                "ticket_title": row["title"],
                "ticket_description": row["description"],
                "subtask_title": subtask["title"],
                "subtask_reasoning": subtask["reasoning"],
                "label": "",
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("evaluation/results.csv"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/hallucination_review.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    review_df = expand_subtasks(df)

    if review_df.empty:
        print("No complex tickets with subtasks in this results file - nothing to review.")
        return

    review_df.to_csv(args.out, index=False)
    print(LABEL_GUIDE)
    print(f"Wrote {len(review_df)} subtask rows to {args.out}")
    print("Open it, fill in the `label` column, then run rq4_hallucination.py.")


if __name__ == "__main__":
    main()
