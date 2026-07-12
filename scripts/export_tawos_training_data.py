"""Export TAWOS training CSVs from MySQL.

Run:  python scripts/export_tawos_training_data.py

Requires MySQL with the tawos database loaded (mysql tawos < TAWOS.sql).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

from tawos_data import (
    DEFAULT_DATABASE_URL,
    balanced_story_point_sample,
    issues_to_export_df,
    load_issues_for_export,
    story_point_distribution,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FULL_CSV = DATA_DIR / "tawos_with_story_points.csv"
BALANCED_CSV = DATA_DIR / "tawos_balanced_train.csv"
BALANCED_WITH_ZERO_CSV = DATA_DIR / "tawos_balanced_train_with_zero.csv"


def _print_distribution(title: str, df) -> None:
    print(f"\n{title}")
    distribution = story_point_distribution(df)
    print(distribution.to_string(index=False))


def main() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    try:
        positive_raw = load_issues_for_export(database_url, include_zero=False)
        with_zero_raw = load_issues_for_export(database_url, include_zero=True)
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

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    full_df = issues_to_export_df(positive_raw)
    balanced_df = balanced_story_point_sample(positive_raw, include_zero=False)
    balanced_with_zero_df = balanced_story_point_sample(with_zero_raw, include_zero=True)

    full_df.to_csv(FULL_CSV, index=False)
    balanced_df.to_csv(BALANCED_CSV, index=False)
    balanced_with_zero_df.to_csv(BALANCED_WITH_ZERO_CSV, index=False)

    print("=== TAWOS training CSV export ===\n")
    print(f"Positive pool (source):     {len(positive_raw):,} tickets")
    print(f"Zero-inclusive pool:        {len(with_zero_raw):,} tickets")
    print(f"Wrote full dataset:         {FULL_CSV} ({len(full_df):,} rows)")
    print(f"Wrote balanced train:       {BALANCED_CSV} ({len(balanced_df):,} rows)")
    print(
        "Wrote balanced with zero:   "
        f"{BALANCED_WITH_ZERO_CSV} ({len(balanced_with_zero_df):,} rows)"
    )

    _print_distribution("Full dataset story point distribution:", full_df)
    _print_distribution("Balanced train story point distribution:", balanced_df)
    _print_distribution(
        "Balanced train (with zero) story point distribution:",
        balanced_with_zero_df,
    )


if __name__ == "__main__":
    main()
