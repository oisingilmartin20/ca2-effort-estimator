"""Unit tests for RAG story-point computation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from estimator import compute_rag_story_points, snap_to_fibonacci
from similarity_search import SimilarTicket


def _ticket(key: str, sp: int, sim: float) -> SimilarTicket:
    return SimilarTicket(
        issue_key=key,
        title="Test",
        description="desc",
        story_points=sp,
        similarity=sim,
    )


def test_single_neighbour_returns_snapped_sp():
    sp, raw, conf = compute_rag_story_points([_ticket("A-1", 5, 0.9)])
    assert sp == 5
    assert raw == 5.0
    assert 0.3 <= conf <= 0.9


def test_equal_similarity_averages_then_snaps():
    tickets = [_ticket("A-1", 3, 0.8), _ticket("A-2", 5, 0.8)]
    sp, raw, _ = compute_rag_story_points(tickets)
    assert raw == 4.0
    assert sp == snap_to_fibonacci(4.0)  # 4 -> 5 (higher bracket)


def test_higher_similarity_weights_more():
    low_weight = compute_rag_story_points([
        _ticket("A-1", 3, 0.05),
        _ticket("A-2", 21, 0.95),
    ])
    high_weight = compute_rag_story_points([
        _ticket("A-1", 3, 0.95),
        _ticket("A-2", 21, 0.05),
    ])
    assert low_weight[0] > high_weight[0]


def test_wide_spread_lowers_confidence():
    tight = compute_rag_story_points([
        _ticket("A-1", 5, 0.85),
        _ticket("A-2", 5, 0.8),
    ])
    wide = compute_rag_story_points([
        _ticket("A-1", 1, 0.85),
        _ticket("A-2", 21, 0.8),
    ])
    assert wide[2] < tight[2]


def test_snap_to_fibonacci_higher_bracket():
    assert snap_to_fibonacci(4) == 5
    assert snap_to_fibonacci(10) == 13
