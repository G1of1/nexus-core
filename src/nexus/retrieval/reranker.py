"""Reranking implementations."""

from nexus.models.search import SearchResult
from nexus.providers.base import RerankerProvider


class ScoreReranker(RerankerProvider):
    """Simple reranker that sorts by existing similarity scores."""

    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        return sorted_results[:top_k]
