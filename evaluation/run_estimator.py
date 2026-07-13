"""Runs the estimator over a stratified TAWOS sample and dumps the results for RQ3/RQ4."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from estimator import estimate_ticket

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tawos_balanced_train.csv"

FIELDNAMES = [
    "issue_key", "project", "issue_type", "title", "description",
    "actual_story_points", "predicted_story_points", "confidence",
    "complex_flag", "subtask_count", "subtask_points_sum", "subtask_titles",
    "subtasks_json", "reasoning", "source",
]


def stratified_sample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in df.groupby("actual_story_points")
    ]
    return pd.concat(parts, ignore_index=True)


def run(sample: pd.DataFrame, out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, row in sample.iterrows():
            ticket = {
                "project": row["project"],
                "issue_type": row["issue_type"],
                "title": row["title"],
                "description": row["description"],
            }
            est = estimate_ticket(ticket)

            writer.writerow({
                "issue_key": row["issue_key"],
                "project": row["project"],
                "issue_type": row["issue_type"],
                "title": row["title"],
                "description": row["description"],
                "actual_story_points": row["actual_story_points"],
                "predicted_story_points": est.story_points,
                "confidence": est.confidence,
                "complex_flag": est.complex,
                "subtask_count": len(est.subtasks),
                "subtask_points_sum": sum(s.story_points for s in est.subtasks),
                "subtask_titles": " | ".join(s.title for s in est.subtasks),
                "subtasks_json": json.dumps([
                    {"title": s.title, "story_points": s.story_points, "reasoning": s.reasoning}
                    for s in est.subtasks
                ]),
                "reasoning": est.reasoning,
                "source": est.source,
            })
            print(f"[{i + 1}/{len(sample)}] {row['issue_key']} "
                  f"actual={row['actual_story_points']} "
                  f"predicted={est.story_points} complex={est.complex}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=10,
                         help="Tickets sampled per Fibonacci class (default 10 -> 70 total)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("evaluation/results.csv"))
    args = parser.parse_args()

    load_dotenv()
    df = pd.read_csv(DATA_PATH)
    sample = stratified_sample(df, args.per_class, args.seed)
    run(sample, args.out)
    print(f"\nWrote {len(sample)} rows to {args.out}")


if __name__ == "__main__":
    main()
