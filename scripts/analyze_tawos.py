"""Print basic analytics for the TAWOS Issue table in MySQL.

Run:  python scripts/analyze_tawos.py

Requires MySQL with the tawos database loaded (mysql tawos < TAWOS.sql).
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError

from tawos_data import DEFAULT_DATABASE_URL, compute_summary, load_issues, project_summary


def main() -> None:
    load_dotenv()
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    try:
        df = load_issues(database_url)
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

    summary = compute_summary(df)

    print("=== TAWOS Issue dataset summary ===\n")
    print(f"Total tickets:              {summary['total']:,}")
    print(f"Missing Story_Point:        {summary['missing_story_point']:,}")
    print(f"Missing Title:              {summary['missing_title']:,}")
    print(f"Missing Description_Text:   {summary['missing_description']:,}")
    print(f"Tickets with Priority:      {summary['has_priority']:,}")
    print(f"Missing Priority:           {summary['missing_priority']:,}")

    print("\nDescription_Text length:")
    print(df["Description_Text"].str.len().describe().to_string())

    print("\nPriority value counts:")
    print(df["Priority"].value_counts(dropna=False).to_string())

    print(f"\nUnique Story_Point values:  {summary['unique_story_points']:,}")
    print("\nStory_Point value counts:")
    print(df["Story_Point"].value_counts(dropna=False).to_string())

    print("\n=== Per-project summary ===\n")
    projects = project_summary(df)
    display = projects.copy()
    display["total_tickets"] = display["total_tickets"].map("{:,}".format)
    display["labeled_tickets"] = display["labeled_tickets"].map("{:,}".format)
    display["labeled_pct"] = display["labeled_pct"].map(lambda x: f"{x:.1f}%")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
