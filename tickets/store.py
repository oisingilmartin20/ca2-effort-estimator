"""Load and save user-created tickets to a local JSON file."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

TICKETS_PATH = Path(__file__).resolve().parent.parent / "data" / "user_tickets.json"

TICKET_COLUMNS = ["issue_key", "project", "issue_type", "title", "description"]

_KEY_PATTERN = re.compile(r"^TKT-(\d+)$")


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TICKET_COLUMNS)


def _read_records() -> list[dict[str, str]]:
    if not TICKETS_PATH.exists():
        return []
    with TICKETS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {TICKETS_PATH}")
    return data


def _write_records(records: list[dict[str, str]]) -> None:
    TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = TICKETS_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")
    tmp_path.replace(TICKETS_PATH)


def _next_issue_key(records: list[dict[str, str]]) -> str:
    max_num = 0
    for record in records:
        match = _KEY_PATTERN.match(str(record.get("issue_key", "")))
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"TKT-{max_num + 1:03d}"


def load_tickets() -> pd.DataFrame:
    records = _read_records()
    if not records:
        return _empty_df()
    return pd.DataFrame.from_records(records, columns=TICKET_COLUMNS)


def add_ticket(
    project: str,
    issue_type: str,
    title: str,
    description: str,
) -> str:
    records = _read_records()
    issue_key = _next_issue_key(records)
    records.append({
        "issue_key": issue_key,
        "project": project.strip(),
        "issue_type": issue_type.strip(),
        "title": title.strip(),
        "description": description.strip(),
    })
    _write_records(records)
    return issue_key
