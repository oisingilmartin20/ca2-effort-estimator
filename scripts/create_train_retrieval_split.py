"""Create 80/20 retrieval corpus and training holdout CSVs from TAWOS.

Run:  python scripts/create_train_retrieval_split.py

Requires MySQL with the tawos database loaded (mysql tawos < TAWOS.sql).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sqlalchemy.exc import SQLAlchemyError

from tawos_data import (
    DEFAULT_DATABASE_URL,
    issues_to_export_df,
    load_issues_for_export,
    story_point_distribution,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_RETRIEVAL_CSV = DATA_DIR / "tawos_retrieval_corpus.csv"
DEFAULT_TRAIN_CSV = DATA_DIR / "tawos_train_holdout.csv"


def _eligible_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a non-empty description and a non-negative story-point label."""
    working = df.copy()
    working["description"] = working["description"].fillna("").astype(str).str.strip()
    working["actual_story_points"] = pd.to_numeric(
        working["actual_story_points"], errors="coerce"
    )
    mask = (
        (working["description"] != "")
        & working["actual_story_points"].notna()
        & (working["actual_story_points"] >= 0)
    )
    return working.loc[mask].reset_index(drop=True)


def _stratify_labels(df: pd.DataFrame) -> pd.Series | None:
    """Return stratification labels when every class has at least two rows."""
    labels = df["actual_story_points"].astype(int)
    counts = labels.value_counts()
    if (counts < 2).any():
        return None
    return labels


def create_split(
    database_url: str,
    *,
    retrieval_out: Path,
    train_out: Path,
    fraction: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TAWOS issues, filter, and split into retrieval/train CSVs."""
    raw = load_issues_for_export(database_url, include_zero=True)
    exported = issues_to_export_df(raw)
    pool = _eligible_pool(exported)

    if pool.empty:
        raise ValueError("No eligible tickets with description and story points found.")

    stratify = _stratify_labels(pool)
    retrieval_df, train_df = train_test_split(
        pool,
        test_size=1 - fraction,
        random_state=seed,
        stratify=stratify,
    )

    retrieval_out.parent.mkdir(parents=True, exist_ok=True)
    retrieval_df = retrieval_df.sort_values("issue_key").reset_index(drop=True)
    train_df = train_df.sort_values("issue_key").reset_index(drop=True)

    retrieval_df.to_csv(retrieval_out, index=False)
    train_df.to_csv(train_out, index=False)
    return retrieval_df, train_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split TAWOS tickets into retrieval corpus (80%) and training holdout (20%)."
    )
    parser.add_argument("--database-url", default=None, help="MySQL SQLAlchemy URL")
    parser.add_argument(
        "--retrieval-out",
        type=Path,
        default=DEFAULT_RETRIEVAL_CSV,
        help="Output path for the retrieval corpus CSV",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Output path for the training holdout CSV",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.8,
        help="Fraction of eligible tickets for the retrieval corpus (default 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    args = parser.parse_args()

    load_dotenv()
    database_url = args.database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    try:
        retrieval_df, train_df = create_split(
            database_url,
            retrieval_out=args.retrieval_out,
            train_out=args.train_out,
            fraction=args.fraction,
            seed=args.seed,
        )
    except SQLAlchemyError as exc:
        print("Error: could not connect to MySQL or read from Issue table.", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print(
            "\nEnsure MySQL is running and the tawos database is loaded:\n"
            "  mysql tawos < TAWOS.sql\n"
            f"Connection string: {database_url}",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    total = len(retrieval_df) + len(train_df)
    retrieval_pct = len(retrieval_df) / total * 100 if total else 0.0
    train_pct = len(train_df) / total * 100 if total else 0.0

    print("=== TAWOS retrieval / training split ===\n")
    print(f"Eligible tickets:           {total:,}")
    print(f"Wrote retrieval corpus:     {args.retrieval_out} ({len(retrieval_df):,} rows, {retrieval_pct:.1f}%)")
    print(f"Wrote training holdout:     {args.train_out} ({len(train_df):,} rows, {train_pct:.1f}%)")

    print("\nRetrieval corpus story point distribution:")
    print(story_point_distribution(retrieval_df).to_string(index=False))
    print("\nTraining holdout story point distribution:")
    print(story_point_distribution(train_df).to_string(index=False))


if __name__ == "__main__":
    main()
