"""Runs the estimator over a stratified TAWOS sample and dumps results for RQ evals.

Default sample is the training holdout (fair RAG eval against the retrieval corpus).
Use --no-rag for a forced LLM-only ablation on the same tickets.
"""
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATA_PATH = DATA_DIR / "tawos_train_holdout.csv"

FIELDNAMES = [
    "issue_key", "project", "issue_type", "title", "description",
    "actual_story_points", "predicted_story_points", "confidence",
    "complex_flag", "subtask_count", "subtask_points_sum", "subtask_titles",
    "subtasks_json", "reasoning", "source",
    "rag_story_points", "rag_raw_average", "rag_confidence", "no_rag_fallback",
    "similar_tickets_json",
]


def stratified_sample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = [
        group.sample(n=min(per_class, len(group)), random_state=seed)
        for _, group in df.groupby("actual_story_points")
    ]
    return pd.concat(parts, ignore_index=True)


def _similar_tickets_json(est) -> str:
    return json.dumps([
        {
            "issue_key": t.issue_key,
            "story_points": t.story_points,
            "similarity": t.similarity,
            "title": t.title,
        }
        for t in est.similar_tickets
    ])


def run(sample: pd.DataFrame, out_path: Path, *, use_rag: bool) -> None:
    failures: list[str] = []
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
            try:
                est = estimate_ticket(ticket, use_rag=use_rag)
            except Exception as exc:
                failures.append(f"{row['issue_key']}: {exc}")
                print(f"[{i + 1}/{len(sample)}] {row['issue_key']} FAILED: {exc}",
                      flush=True)
                continue

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
                "rag_story_points": est.rag_story_points if est.rag_story_points is not None else "",
                "rag_raw_average": est.rag_raw_average if est.rag_raw_average is not None else "",
                "rag_confidence": est.rag_confidence if est.rag_confidence is not None else "",
                "no_rag_fallback": est.no_rag_fallback,
                "similar_tickets_json": _similar_tickets_json(est),
            })
            f.flush()
            print(f"[{i + 1}/{len(sample)}] {row['issue_key']} "
                  f"actual={row['actual_story_points']} "
                  f"predicted={est.story_points} complex={est.complex} "
                  f"source={est.source}", flush=True)

    if failures:
        print(f"\n{len(failures)} ticket(s) failed:", flush=True)
        for line in failures:
            print(f"  - {line}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run the estimator for RQ evaluation CSVs.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Ticket CSV to sample from (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=10,
        help="Tickets sampled per Fibonacci class (default 10)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/results_rag.csv"),
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Force LLM-only estimation (skip retrieval) for ablation",
    )
    args = parser.parse_args()

    load_dotenv()
    df = pd.read_csv(args.data)
    sample = stratified_sample(df, args.per_class, args.seed)
    run(sample, args.out, use_rag=not args.no_rag)
    print(f"\nWrote {len(sample)} rows to {args.out} "
          f"(use_rag={not args.no_rag})")


if __name__ == "__main__":
    main()
