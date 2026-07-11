"""Shared TAWOS Issue table loader and summary helpers."""
from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

DEFAULT_DATABASE_URL = "mysql+pymysql://root@127.0.0.1/tawos"

ISSUE_QUERY = """
SELECT
  i.Story_Point,
  i.Title,
  i.Description,
  i.Description_Text,
  i.Priority,
  i.Project_ID,
  p.Name AS Project_Name
FROM Issue i
LEFT JOIN Project p ON i.Project_ID = p.ID
"""


def _missing_text(series: pd.Series) -> int:
    filled = series.fillna("")
    return int((filled.str.strip() == "").sum())


def _has_text(series: pd.Series) -> int:
    filled = series.fillna("")
    return int((filled.str.strip() != "").sum())


def load_issues(database_url: str | None = None) -> pd.DataFrame:
    """Load the full Issue table from MySQL."""
    load_dotenv()
    url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    engine = create_engine(url)
    return pd.read_sql(ISSUE_QUERY, engine)


def compute_summary(df: pd.DataFrame) -> dict[str, int]:
    """Return core dataset counts used by the CLI and notebook."""
    total = len(df)
    has_priority = _has_text(df["Priority"])
    return {
        "total": total,
        "missing_story_point": int(df["Story_Point"].isna().sum()),
        "missing_title": _missing_text(df["Title"]),
        "missing_description": _missing_text(df["Description_Text"]),
        "has_priority": has_priority,
        "missing_priority": total - has_priority,
        "unique_story_points": int(df["Story_Point"].nunique(dropna=True)),
    }


def project_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return ticket and labeled story-point counts per project."""
    labeled = df["Story_Point"].notna()
    grouped = (
        df.assign(_labeled=labeled)
        .groupby("Project_Name", dropna=False)
        .agg(total_tickets=("Story_Point", "size"), labeled_tickets=("_labeled", "sum"))
        .reset_index()
    )
    grouped["Project_Name"] = grouped["Project_Name"].fillna("Unknown")
    grouped["labeled_pct"] = (
        grouped["labeled_tickets"] / grouped["total_tickets"] * 100
    ).round(1)
    return grouped.sort_values("total_tickets", ascending=False).reset_index(drop=True)


def _story_point_label(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    numeric = float(value)
    if numeric == int(numeric):
        return str(int(numeric))
    return str(value)


def story_point_counts_from_series(story_points: pd.Series) -> pd.DataFrame:
    """Return sorted story point value counts for a series."""
    counts = story_points.value_counts(dropna=False).reset_index()
    counts.columns = ["story_point", "count"]
    counts["story_point_label"] = counts["story_point"].apply(_story_point_label)
    numeric = pd.to_numeric(counts["story_point"], errors="coerce")
    counts["sort_key"] = numeric.fillna(-1)
    return counts.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def story_point_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return story point value counts sorted for display."""
    return story_point_counts_from_series(df["Story_Point"])


def story_point_counts_by_project(df: pd.DataFrame, project_name: str) -> pd.DataFrame:
    """Return rounded story point counts for a single project."""
    project_df = df[df["Project_Name"] == project_name]
    return rounded_story_point_counts(project_df)


def top_projects_by_labeled_tickets(df: pd.DataFrame, n: int = 5) -> list[str]:
    """Return project names with the most labeled story-point tickets."""
    summary = project_summary(df)
    return summary.nlargest(n, "labeled_tickets")["Project_Name"].tolist()


def rounded_story_point_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return story point counts after rounding; does not modify df."""
    return story_point_counts_from_series(df["Story_Point"].round())


def fractional_story_point_stats(df: pd.DataFrame) -> dict[str, int]:
    """Count tickets with non-null fractional story points and unique values."""
    story_points = df["Story_Point"]
    present = story_points.dropna()
    fractional = present[present != present.round()]
    return {
        "fractional_story_points": int(len(fractional)),
        "unique_raw": int(story_points.nunique(dropna=True)),
        "unique_rounded": int(story_points.round().nunique(dropna=True)),
    }
