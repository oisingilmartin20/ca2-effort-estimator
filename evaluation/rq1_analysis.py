"""RQ1- LLM effort estimate closeness to real story points.

Assumes the LLM was prompted to return values on the Fibonacci scale
(1, 2, 3, 5, 8, 13, 21) directly, so no snapping/rounding is needed.
A validation step checks whether the LLM stayed on-scale.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

FIBONACCI_SCALE = [1, 2, 3, 5, 8, 13, 21]


def validate_fibonacci(df: pd.DataFrame) -> None:
    valid = set(FIBONACCI_SCALE)
    off_scale = df[~df["predicted_story_points"].isin(valid)]
    if off_scale.empty:
        print("Fibonacci validation: all predictions are valid scale values.\n")
    else:
        print(f"WARNING: {len(off_scale)} prediction(s) are off-scale and may skew results:")
        print(off_scale[["issue_key", "predicted_story_points"]].to_string(index=False))
        print()


def error_metrics(df: pd.DataFrame) -> dict[str, float]:
    actual = df["actual_story_points"]
    predicted = df["predicted_story_points"]

    abs_error = (predicted - actual).abs()
    rel_error = abs_error / actual.replace(0, pd.NA)

    mae  = abs_error.mean()
    rmse = ((predicted - actual) ** 2).mean() ** 0.5
    mmre = rel_error.mean()

    pred25 = (rel_error <= 0.25).mean()
    pred50 = (rel_error <= 0.50).mean()

    exact_match = (predicted == actual).mean()

    rho, p_value = spearmanr(actual, predicted)

    return {
        "n":                 len(df),
        "mae":               mae,
        "rmse":              rmse,
        "mmre":              mmre,
        "pred25":            pred25,
        "pred50":            pred50,
        "exact_match_rate":  exact_match,
        "spearman_rho":      rho,
        "spearman_p":        p_value,
    }


def worst_misses(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    out = df.copy()
    out["abs_error"] = (out["predicted_story_points"] - out["actual_story_points"]).abs()
    return out.sort_values("abs_error", ascending=False).head(n)[
        ["issue_key", "actual_story_points", "predicted_story_points", "abs_error"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("evaluation/results_rag.csv"))
    parser.add_argument("--worst-n",  type=int,  default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.results)

    required = {"issue_key", "actual_story_points", "predicted_story_points"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: results.csv is missing column(s): {missing}\n"
                         f"Found: {list(df.columns)}")

    validate_fibonacci(df)

    metrics = error_metrics(df)

    print("=== RQ1: LLM estimate closeness to actual story points ===\n")
    print(f"Total tickets evaluated:  {metrics['n']}")
    print(f"MAE:                      {metrics['mae']:.2f} points")
    print(f"RMSE:                     {metrics['rmse']:.2f} points")
    print(f"MMRE:                     {metrics['mmre']:.2f}")
    print(f"PRED(25):                 {metrics['pred25']:.1%}")
    print(f"PRED(50):                 {metrics['pred50']:.1%}")
    print(f"Exact match rate:         {metrics['exact_match_rate']:.1%}")
    print(f"Spearman rho:             {metrics['spearman_rho']:.2f}  "
          f"(p={metrics['spearman_p']:.3f})")

    print(f"\n=== {args.worst_n} worst misses ===")
    print(worst_misses(df, args.worst_n).to_string(index=False))


if __name__ == "__main__":
    main()