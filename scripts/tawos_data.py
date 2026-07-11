"""Shared TAWOS Issue table loader and summary helpers."""
from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

DEFAULT_DATABASE_URL = "mysql+pymysql://root@127.0.0.1/tawos"
MAX_STORY_POINT = 100
# Mirrors estimator.FIBONACCI for training label alignment.
FIBONACCI = [1, 2, 3, 5, 8, 13, 21]

ISSUE_QUERY = """
SELECT Story_Point, Title, Description, Description_Text, Priority
FROM Issue
"""

EXPORT_QUERY = """
SELECT
  i.ID,
  i.Issue_Key,
  p.Name AS Project_Name,
  i.Type,
  i.Title,
  i.Description,
  i.Description_Text,
  i.Story_Point
FROM Issue i
LEFT JOIN Project p ON i.Project_ID = p.ID
WHERE i.Story_Point IS NOT NULL
  AND i.Story_Point <= {max_sp}
  {zero_clause}
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


def load_issues_for_export(
    database_url: str | None = None,
    *,
    include_zero: bool = False,
) -> pd.DataFrame:
    """Load Issue rows joined to Project for CSV export."""
    load_dotenv()
    url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    zero_clause = "" if include_zero else "AND i.Story_Point > 0"
    query = EXPORT_QUERY.format(max_sp=MAX_STORY_POINT, zero_clause=zero_clause)
    engine = create_engine(url)
    return pd.read_sql(query, engine)


def to_nearest_fibonacci(value: float) -> int:
    """Map a positive story point to the nearest Fibonacci scale value."""
    best = FIBONACCI[0]
    best_distance = abs(value - best)
    for candidate in FIBONACCI[1:]:
        distance = abs(value - candidate)
        if distance < best_distance or (
            distance == best_distance and candidate < best
        ):
            best = candidate
            best_distance = distance
    return best


def story_point_sample_class(story_point: float, *, include_zero: bool) -> int:
    """Return the class label used for balanced sampling."""
    if include_zero and story_point == 0:
        return 0
    return to_nearest_fibonacci(story_point)


def _pick_description(row: pd.Series) -> str:
    text = row.get("Description_Text")
    if pd.notna(text) and str(text).strip():
        return str(text)
    description = row.get("Description")
    return "" if pd.isna(description) else str(description)


def issues_to_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw Issue export rows to the tawos_sample.csv schema."""
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        story_point = float(row["Story_Point"])
        if story_point == 0:
            actual_story_points = 0
        else:
            actual_story_points = to_nearest_fibonacci(story_point)

        issue_key = row.get("Issue_Key")
        if pd.isna(issue_key) or not str(issue_key).strip():
            issue_key = f"TAWOS-{int(row['ID'])}"
        else:
            issue_key = str(issue_key)

        project = row.get("Project_Name")
        project_name = "Unknown" if pd.isna(project) or not str(project).strip() else str(project)

        title = row.get("Title")
        title_text = "" if pd.isna(title) else str(title)

        issue_type = row.get("Type")
        issue_type_text = "" if pd.isna(issue_type) else str(issue_type)

        records.append(
            {
                "issue_key": issue_key,
                "project": project_name,
                "issue_type": issue_type_text,
                "title": title_text,
                "description": _pick_description(row),
                "actual_story_points": actual_story_points,
            }
        )

    return pd.DataFrame.from_records(records)


def balanced_story_point_sample(
    df: pd.DataFrame,
    *,
    fraction: float = 0.2,
    include_zero: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a balanced random subset grouped by zero/Fibonacci story point class."""
    if df.empty:
        return df.copy()

    working = df.copy()
    working["_sample_class"] = working["Story_Point"].map(
        lambda value: story_point_sample_class(float(value), include_zero=include_zero)
    )

    target_n = round(len(working) * fraction)
    present_classes = sorted(working["_sample_class"].unique())
    n_per_class = target_n // len(present_classes) if present_classes else 0

    sampled_parts: list[pd.DataFrame] = []
    for sample_class in present_classes:
        group = working[working["_sample_class"] == sample_class]
        sample_size = min(n_per_class, len(group))
        if sample_size > 0:
            sampled_parts.append(group.sample(n=sample_size, random_state=random_state))

    if not sampled_parts:
        return issues_to_export_df(df.iloc[0:0])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return issues_to_export_df(sampled)


def story_point_distribution(df: pd.DataFrame, label_col: str = "actual_story_points") -> pd.DataFrame:
    """Return sorted counts for a story-point label column."""
    counts = df[label_col].value_counts(dropna=False).reset_index()
    counts.columns = [label_col, "count"]
    return counts.sort_values(label_col).reset_index(drop=True)


def compute_summary(df: pd.DataFrame) -> dict[str, int]:
    """Return core dataset counts used by the CLI and notebook."""
    total = len(df)
    has_priority = _has_text(df["Priority"])
    return {
        "total": total,
        "missing_story_point": int(df["Story_Point"].isna().sum()),
        "missing_title": _missing_text(df["Title"]),
        "missing_description": _missing_text(df["Description"]),
        "has_priority": has_priority,
        "missing_priority": total - has_priority,
        "unique_story_points": int(df["Story_Point"].nunique(dropna=True)),
    }


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
