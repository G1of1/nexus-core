"""Tests for evaluation metrics."""

import pytest

from nexus.evaluation.metrics import compute_answer_metrics, compute_retrieval_metrics


def test_retrieval_metrics_perfect_precision():
    metrics = compute_retrieval_metrics(
        retrieved_ids=["a", "b", "c"],
        relevant_ids={"a", "b"},
        k=3,
    )
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.hit_rate == 1.0
    assert metrics.mrr == 1.0


def test_retrieval_metrics_no_hits():
    metrics = compute_retrieval_metrics(
        retrieved_ids=["x", "y"],
        relevant_ids={"a", "b"},
        k=2,
    )
    assert metrics.precision_at_k == 0.0
    assert metrics.hit_rate == 0.0
    assert metrics.mrr == 0.0


def test_retrieval_metrics_empty():
    metrics = compute_retrieval_metrics(
        retrieved_ids=[],
        relevant_ids={"a"},
        k=5,
    )
    assert metrics.precision_at_k == 0.0


def test_answer_metrics_with_citations():
    metrics = compute_answer_metrics(
        answer="Based on the context, the answer is 42.",
        citation_count=2,
        expected_keywords=["42"],
    )
    assert metrics.has_citations
    assert metrics.citation_count == 2
    assert metrics.contains_grounding_phrase
    assert metrics.metadata["keyword_recall"] == 1.0


def test_answer_metrics_no_grounding():
    metrics = compute_answer_metrics(
        answer="The answer is definitely 999.",
        citation_count=0,
    )
    assert not metrics.has_citations
    assert not metrics.contains_grounding_phrase
