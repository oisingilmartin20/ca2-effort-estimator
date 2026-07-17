"""RQ4 step 2 - crunch the hallucination rate once the review sheet is labeled."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VALID_LABELS = {"grounded", "inferred", "hallucinated"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=Path("evaluation/hallucination_review_labeled.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.review)
    labeled = df[df["label"].isin(VALID_LABELS)]
    unlabeled = len(df) - len(labeled)

    if unlabeled:
        print(f"Warning: {unlabeled} of {len(df)} subtasks are not yet labeled "
              f"(expected one of {sorted(VALID_LABELS)}). Results below only cover labeled rows.\n")

    if labeled.empty:
        print("No labeled rows yet - fill in the `label` column first.")
        return

    counts = labeled["label"].value_counts()
    total = len(labeled)

    print(f"=== RQ4: hallucination rate ({total} labeled subtasks) ===\n")
    for label in ["grounded", "inferred", "hallucinated"]:
        n = int(counts.get(label, 0))
        print(f"{label:>13}: {n:>4}  ({n / total:.1%})")

    hallucinated_rate = counts.get("hallucinated", 0) / total
    print(f"\nOverall hallucination rate: {hallucinated_rate:.1%}")

    tickets_with_hallucination = (
        labeled[labeled["label"] == "hallucinated"]["issue_key"].nunique()
    )
    total_tickets = labeled["issue_key"].nunique()
    print(f"Tickets with >=1 hallucinated subtask: "
          f"{tickets_with_hallucination}/{total_tickets} "
          f"({tickets_with_hallucination / total_tickets:.1%})")


if __name__ == "__main__":
    main()
