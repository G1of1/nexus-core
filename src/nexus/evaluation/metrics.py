"""Metrics for evaluating RAG pipeline quality."""

from pydantic import BaseModel, Field


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval quality evaluation."""

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    hit_rate: float = 0.0
    k: int = 5


class AnswerMetrics(BaseModel):
    """Metrics for answer quality evaluation."""

    has_citations: bool = False
    citation_count: int = 0
    answer_length: int = 0
    contains_grounding_phrase: bool = False
    metadata: dict = Field(default_factory=dict)


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int | None = None,
) -> RetrievalMetrics:
    """Compute precision@k, recall@k, MRR, and hit rate."""
    k = k or len(retrieved_ids)
    top_k = retrieved_ids[:k]

    if not top_k:
        return RetrievalMetrics(k=k)

    hits = [i for i, doc_id in enumerate(top_k) if doc_id in relevant_ids]
    hit_count = len(hits)

    precision = hit_count / len(top_k) if top_k else 0.0
    recall = hit_count / len(relevant_ids) if relevant_ids else 0.0
    hit_rate = 1.0 if hit_count > 0 else 0.0
    mrr = 1.0 / (hits[0] + 1) if hits else 0.0

    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
        hit_rate=hit_rate,
        k=k,
    )


def compute_answer_metrics(
    answer: str,
    citation_count: int = 0,
    expected_keywords: list[str] | None = None,
) -> AnswerMetrics:
    """Compute basic answer quality metrics."""
    grounding_phrases = [
        "based on the context",
        "according to the documents",
        "the provided context",
        "don't have enough information",
        "do not have enough information",
    ]
    answer_lower = answer.lower()
    has_grounding = any(phrase in answer_lower for phrase in grounding_phrases)

    metadata: dict = {}
    if expected_keywords:
        found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        metadata["keywords_found"] = found
        metadata["keyword_recall"] = len(found) / len(expected_keywords) if expected_keywords else 0.0

    return AnswerMetrics(
        has_citations=citation_count > 0,
        citation_count=citation_count,
        answer_length=len(answer),
        contains_grounding_phrase=has_grounding,
        metadata=metadata,
    )
