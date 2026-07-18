"""RQ2 justification step 2 - automatic citation checks + human rubric rates."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RUBRIC_COLUMNS = [
    "grounded",
    "faithful_to_rag",
    "comparative",
    "no_hallucination",
    "useful",
]

# Common Jira-style keys: PROJECT-123
ISSUE_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")


def _parse_neighbours(raw) -> list[dict]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _neighbour_keys(row: pd.Series) -> set[str]:
    if "similar_tickets_json" in row.index:
        neigh = _parse_neighbours(row["similar_tickets_json"])
        if neigh:
            return {
                str(t.get("issue_key", "")).upper()
                for t in neigh
                if t.get("issue_key")
            }

    raw = str(row.get("neighbour_keys", "") or "")
    return {m.group(1).upper() for m in ISSUE_KEY_RE.finditer(raw)}


def automatic_checks(df: pd.DataFrame) -> dict[str, float]:
    cites_ok = 0
    no_invented = 0
    sp_ok = 0
    eligible_cite = 0
    n = len(df)

    for _, row in df.iterrows():
        reasoning = str(row.get("reasoning", "") or "")
        cited = {m.group(1).upper() for m in ISSUE_KEY_RE.finditer(reasoning)}
        neighbours = _neighbour_keys(row)

        if neighbours:
            eligible_cite += 1
            if cited & neighbours:
                cites_ok += 1

        invented = cited - neighbours
        if not invented:
            no_invented += 1

        sp = row.get("predicted_story_points")
        if sp is None or (isinstance(sp, float) and pd.isna(sp)):
            sp_ok += 1
            continue
        sp_int = int(float(sp))
        contradiction = re.search(
            r"(?:story\s*points?|estimate(?:d)?(?:\s+at)?)\s*[:=]?\s*(\d+)",
            reasoning,
            re.IGNORECASE,
        )
        if contradiction and int(contradiction.group(1)) != sp_int:
            continue
        sp_ok += 1

    return {
        "n": n,
        "pct_cites_neighbour": (
            cites_ok / eligible_cite if eligible_cite else float("nan")
        ),
        "n_eligible_cite": eligible_cite,
        "pct_no_invented_keys": no_invented / n if n else 0.0,
        "pct_sp_consistent": sp_ok / n if n else 0.0,
    }


def human_rubric_rates(df: pd.DataFrame) -> None:
    labeled_mask = pd.Series(True, index=df.index)
    for col in RUBRIC_COLUMNS:
        if col not in df.columns:
            print(f"Missing rubric column `{col}` — cannot score human labels.")
            return
        labeled_mask &= df[col].astype(str).str.strip().isin({"0", "1"})

    labeled = df[labeled_mask]
    unlabeled = len(df) - len(labeled)

    if unlabeled:
        print(
            f"Warning: {unlabeled} of {len(df)} rows are not fully labeled "
            f"(expected 0/1 in {RUBRIC_COLUMNS}). "
            "Rates below cover labeled rows only.\n"
        )

    if labeled.empty:
        print("No labeled rows yet — fill in the rubric columns first.")
        return

    print(f"=== Human rubric rates ({len(labeled)} labeled rows) ===\n")
    scores = []
    for col in RUBRIC_COLUMNS:
        vals = labeled[col].astype(int)
        rate = float(vals.mean())
        scores.append(rate)
        print(f"{col:>18}: {rate:.1%}  ({int(vals.sum())}/{len(labeled)})")

    print(f"\nMean rubric score: {sum(scores) / len(scores):.1%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("evaluation/justification_review.csv"),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.review)
    if df.empty:
        print("Review file is empty.")
        return

    print("=== RQ2: justification faithfulness ===\n")

    auto = automatic_checks(df)
    print("=== Automatic checks ===")
    print(f"Total rows:                    {auto['n']}")
    print(
        f"Cites >=1 retrieved key:       {auto['pct_cites_neighbour']:.1%}  "
        f"(of {auto['n_eligible_cite']} rows with neighbours)"
    )
    print(f"No invented issue keys:        {auto['pct_no_invented_keys']:.1%}")
    print(f"Consistent with fixed SP:      {auto['pct_sp_consistent']:.1%}")
    print()

    human_rubric_rates(df)


if __name__ == "__main__":
    main()
