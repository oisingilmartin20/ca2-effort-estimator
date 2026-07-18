"""RQ2 - RAG-on vs RAG-off ablation and retrieval quality stats.

Compares primary system quality (RAG) against the LLM-only fallback on the
same holdout tickets, and summarises neighbour usefulness on the RAG run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_analysis import error_metrics  # noqa: E402


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"=== {label} (n={metrics['n']}) ===")
    print(f"MAE:              {metrics['mae']:.2f}")
    print(f"RMSE:             {metrics['rmse']:.2f}")
    print(f"MMRE:             {metrics['mmre']:.2f}")
    print(f"PRED(25):         {metrics['pred25']:.1%}")
    print(f"PRED(50):         {metrics['pred50']:.1%}")
    print(f"Exact match:      {metrics['exact_match_rate']:.1%}")
    print(f"Spearman rho:     {metrics['spearman_rho']:.2f}  "
          f"(p={metrics['spearman_p']:.3f})")
    print()


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


def retrieval_stats(df: pd.DataFrame) -> dict[str, float]:
    neighbours = df["similar_tickets_json"].map(_parse_neighbours)
    with_neighbours = neighbours.map(bool)
    n = len(df)
    n_with = int(with_neighbours.sum())

    top1_sims: list[float] = []
    mean_topk_sims: list[float] = []
    spreads: list[float] = []
    same_project_rates: list[float] = []

    for (_, row), neigh in zip(df.iterrows(), neighbours):
        if not neigh:
            continue
        sims = [float(t.get("similarity", 0.0)) for t in neigh]
        sps = [int(t.get("story_points", 0)) for t in neigh]
        top1_sims.append(max(sims) if sims else 0.0)
        mean_topk_sims.append(sum(sims) / len(sims) if sims else 0.0)
        spreads.append(max(sps) - min(sps) if sps else 0.0)

        keys = [str(t.get("issue_key", "")) for t in neigh]
        if keys:
            # Same project-key stem (e.g. MESOS-, DM-) as a relevance proxy
            stem = str(row["issue_key"]).split("-")[0]
            stem_hits = sum(1 for k in keys if k.split("-")[0] == stem)
            same_project_rates.append(stem_hits / len(keys))

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "n": n,
        "pct_with_neighbours": n_with / n if n else 0.0,
        "mean_top1_similarity": _mean(top1_sims),
        "mean_topk_similarity": _mean(mean_topk_sims),
        "mean_neighbour_sp_spread": _mean(spreads),
        "mean_same_project_rate": _mean(same_project_rates),
    }


def paired_comparison(rag_df: pd.DataFrame, norag_df: pd.DataFrame) -> dict[str, float]:
    merged = rag_df.merge(
        norag_df,
        on="issue_key",
        suffixes=("_rag", "_norag"),
    )
    if merged.empty:
        return {
            "n_paired": 0,
            "mean_abs_error_delta": float("nan"),
            "pct_rag_closer": float("nan"),
        }

    err_rag = (merged["predicted_story_points_rag"] - merged["actual_story_points_rag"]).abs()
    err_norag = (merged["predicted_story_points_norag"] - merged["actual_story_points_norag"]).abs()
    delta = err_rag - err_norag  # negative => RAG better
    closer = err_rag < err_norag

    return {
        "n_paired": len(merged),
        "mean_abs_error_delta": float(delta.mean()),
        "pct_rag_closer": float(closer.mean()),
    }


def confidence_band_mae(df: pd.DataFrame) -> None:
    if "rag_confidence" not in df.columns:
        print("No rag_confidence column — skipping confidence bands.\n")
        return

    conf = pd.to_numeric(df["rag_confidence"], errors="coerce")
    working = df.copy()
    working["_conf"] = conf
    working = working[working["_conf"].notna()]
    if working.empty:
        print("No numeric rag_confidence values — skipping confidence bands.\n")
        return

    high = working[working["_conf"] >= 0.7]
    low = working[working["_conf"] < 0.55]

    def _mae(subset: pd.DataFrame) -> float:
        if subset.empty:
            return float("nan")
        return float(
            (subset["predicted_story_points"] - subset["actual_story_points"]).abs().mean()
        )

    print("=== RAG confidence bands (MAE) ===")
    print(f"High (>=0.70): n={len(high)}  MAE={_mae(high):.2f}")
    print(f"Low  (<0.55):  n={len(low)}  MAE={_mae(low):.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", type=Path, default=Path("evaluation/results_rag.csv"))
    parser.add_argument("--norag", type=Path, default=Path("evaluation/results_norag.csv"))
    args = parser.parse_args()

    required = {"issue_key", "actual_story_points", "predicted_story_points"}

    rag_df = pd.read_csv(args.rag)
    norag_df = pd.read_csv(args.norag)

    for label, frame, path in (
        ("RAG", rag_df, args.rag),
        ("no-RAG", norag_df, args.norag),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise SystemExit(f"ERROR: {path} missing column(s): {missing}")

    print("=== RQ2: RAG-on vs RAG-off ablation ===\n")
    _print_metrics("RAG-on (primary)", error_metrics(rag_df))
    _print_metrics("RAG-off (fallback baseline)", error_metrics(norag_df))

    paired = paired_comparison(rag_df, norag_df)
    print("=== Paired comparison (same issue_key) ===")
    print(f"Paired tickets:           {paired['n_paired']}")
    print(f"Mean |error| delta (RAG - no-RAG): {paired['mean_abs_error_delta']:.2f}  "
          "(negative => RAG closer on average)")
    print(f"% tickets where RAG closer: {paired['pct_rag_closer']:.1%}")
    print()

    if "similar_tickets_json" in rag_df.columns:
        stats = retrieval_stats(rag_df)
        print("=== Retrieval stats (RAG run) ===")
        print(f"% with neighbours:        {stats['pct_with_neighbours']:.1%}")
        print(f"Mean top-1 similarity:    {stats['mean_top1_similarity']:.3f}")
        print(f"Mean top-k similarity:    {stats['mean_topk_similarity']:.3f}")
        print(f"Mean neighbour SP spread: {stats['mean_neighbour_sp_spread']:.2f}")
        print(f"Mean same-project rate:   {stats['mean_same_project_rate']:.1%}")
        print()
    else:
        print("No similar_tickets_json column on RAG file — skipping retrieval stats.\n")

    confidence_band_mae(rag_df)


if __name__ == "__main__":
    main()
