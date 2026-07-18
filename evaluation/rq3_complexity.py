"""RQ3 - complexity detection and subtask reconciliation.

TAWOS has no "should this have been decomposed" label, so we treat
actual_story_points >= COMPLEX_THRESHOLD as ground truth. Team agreed on 8.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COMPLEX_THRESHOLD = 8


def confusion_counts(df: pd.DataFrame, threshold: int) -> dict[str, int]:
    actual_complex = df["actual_story_points"] >= threshold
    predicted_complex = df["complex_flag"].astype(bool)

    return {
        "true_positive": int((actual_complex & predicted_complex).sum()),
        "false_positive": int((~actual_complex & predicted_complex).sum()),
        "false_negative": int((actual_complex & ~predicted_complex).sum()),
        "true_negative": int((~actual_complex & ~predicted_complex).sum()),
    }


def precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["true_positive"], counts["false_positive"], counts["false_negative"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def reconciliation_gap(df: pd.DataFrame) -> pd.DataFrame:
    complex_rows = df[df["complex_flag"].astype(bool) & (df["subtask_count"] > 0)].copy()
    complex_rows["gap"] = complex_rows["subtask_points_sum"] - complex_rows["predicted_story_points"]
    return complex_rows[[
        "issue_key", "predicted_story_points", "subtask_points_sum", "gap", "subtask_titles",
    ]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("evaluation/results_rag.csv"))
    parser.add_argument("--threshold", type=int, default=COMPLEX_THRESHOLD)
    args = parser.parse_args()

    df = pd.read_csv(args.results)

    counts = confusion_counts(df, args.threshold)
    metrics = precision_recall_f1(counts)

    print(f"=== RQ3: complexity detection (ground truth: actual_sp >= {args.threshold}) ===\n")
    print(f"Total tickets evaluated: {len(df)}")
    print(f"Confusion counts: {counts}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1:        {metrics['f1']:.2f}")

    gaps = reconciliation_gap(df)
    print(f"\n=== Subtask point reconciliation ({len(gaps)} decomposed tickets) ===")
    if not gaps.empty:
        print(f"Mean gap (subtask sum - top-level estimate): {gaps['gap'].mean():.2f}")
        print(gaps.to_string(index=False))
    else:
        print("No tickets were flagged complex with subtasks in this sample.")


if __name__ == "__main__":
    main()
