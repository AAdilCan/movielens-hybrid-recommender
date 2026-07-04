"""Tests for the ranking metrics against hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.eval.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# Ranked recommendations; relevant items are {1, 3, 5}.
RECS = [1, 2, 3, 4, 5]
RELEVANT = {1, 3, 5}


def test_precision_at_k():
    # Top-3 = [1,2,3]; hits at positions 1 and 3 -> 2/3.
    assert precision_at_k(RECS, RELEVANT, 3) == pytest.approx(2 / 3)
    # Top-5 has all 3 hits -> 3/5.
    assert precision_at_k(RECS, RELEVANT, 5) == pytest.approx(3 / 5)


def test_recall_at_k():
    # Top-3 captures 2 of 3 relevant.
    assert recall_at_k(RECS, RELEVANT, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RECS, RELEVANT, 5) == pytest.approx(1.0)


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(RECS, set(), 5) == 0.0


def test_average_precision():
    # Hits at ranks 1, 3, 5. Precision at each hit: 1/1, 2/3, 3/5.
    # AP = (1 + 2/3 + 3/5) / min(5, 3) = (1 + 0.6667 + 0.6) / 3.
    expected = (1.0 + 2 / 3 + 3 / 5) / 3
    assert average_precision_at_k(RECS, RELEVANT, 5) == pytest.approx(expected)


def test_average_precision_perfect_ranking():
    # All relevant items ranked first -> AP = 1.0.
    assert average_precision_at_k([1, 3, 5, 2, 4], {1, 3, 5}, 5) == pytest.approx(1.0)


def test_ndcg_perfect_is_one():
    assert ndcg_at_k([1, 3, 5, 2, 4], {1, 3, 5}, 5) == pytest.approx(1.0)


def test_ndcg_matches_manual_dcg():
    # Gains at ranks 1,3,5 -> DCG = 1/log2(2) + 1/log2(4) + 1/log2(6).
    dcg = 1 / np.log2(2) + 1 / np.log2(4) + 1 / np.log2(6)
    idcg = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(RECS, RELEVANT, 5) == pytest.approx(dcg / idcg)


def test_ndcg_zero_when_no_hits():
    assert ndcg_at_k([2, 4, 6], {1, 3, 5}, 3) == 0.0


def test_hit_rate():
    assert hit_rate_at_k(RECS, RELEVANT, 2) == 1.0  # rank 1 is relevant
    assert hit_rate_at_k([2, 4], RELEVANT, 2) == 0.0


def test_metrics_handle_short_lists():
    # Fewer recommendations than k should not error.
    assert precision_at_k([1], {1}, 5) == pytest.approx(1 / 5)
    assert recall_at_k([1], {1, 2}, 5) == pytest.approx(1 / 2)
