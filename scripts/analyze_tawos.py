"""Print basic analytics for the TAWOS Issue table in MySQL.

Run:  python scripts/analyze_tawos.py

Requires MySQL with the tawos database loaded (mysql tawos < TAWOS.sql).
"""
from __future__ import annotations

import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

DEFAULT_DATABASE_URL = "mysql+pymysql://root@127.0.0.1/tawos"

ISSUE_QUERY = """
SELECT Story_Point, Title, Description, Description_Text, Priority
FROM Issue
"""


def _missing_text(series: pd.Series) -> int:
    filled = series.fillna("")
    return int((filled.str.strip() == "").sum())


def _has_text(series: pd.Series) -> int:
    filled = series.fillna("")
    return int((filled.str.strip() != "").sum())


def main() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    try:
        engine = create_engine(database_url)
        df = pd.read_sql(ISSUE_QUERY, engine)
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

    total = len(df)
    missing_story_point = int(df["Story_Point"].isna().sum())
    missing_title = _missing_text(df["Title"])
    missing_description = _missing_text(df["Description"])
    has_priority = _has_text(df["Priority"])
    missing_priority = total - has_priority
    unique_story_points = df["Story_Point"].nunique(dropna=True)

    print("=== TAWOS Issue dataset summary ===\n")
    print(f"Total tickets:              {total:,}")
    print(f"Missing Story_Point:        {missing_story_point:,}")
    print(f"Missing Title:              {missing_title:,}")
    print(f"Missing Description:        {missing_description:,}")
    print(f"Tickets with Priority:      {has_priority:,}")
    print(f"Missing Priority:           {missing_priority:,}")

    print("\nDescription_Text length:")
    print(df["Description_Text"].str.len().describe().to_string())

    print("\nPriority value counts:")
    print(df["Priority"].value_counts(dropna=False).to_string())

    print(f"\nUnique Story_Point values:  {unique_story_points:,}")
    print("\nStory_Point value counts:")
    print(df["Story_Point"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
