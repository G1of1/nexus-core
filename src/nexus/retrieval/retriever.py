"""Core retrieval service combining embedding and vector search."""

from nexus.config import NexusSettings
from nexus.exceptions import RetrievalError
from nexus.models.search import SearchFilter, SearchQuery, SearchResult
from nexus.providers.base import EmbeddingProvider, RerankerProvider, VectorStore


class Retriever:
    """Orchestrates query embedding, vector search, and optional reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        settings: NexusSettings | None = None,
        reranker: RerankerProvider | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._settings = settings
        self._reranker = reranker
        self._default_top_k = settings.top_k if settings else 5
        self._default_rerank_top_k = settings.rerank_top_k if settings else 3
        self._default_score_threshold = settings.score_threshold if settings else None

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: list[SearchFilter] | None = None,
        score_threshold: float | None = None,
        rerank: bool = True,
    ) -> list[SearchResult]:
        top_k = top_k or self._default_top_k
        score_threshold = (
            score_threshold if score_threshold is not None else self._default_score_threshold
        )

        try:
            query_vector = await self._embedding_provider.embed_query(query)
        except Exception as e:
            raise RetrievalError(f"Failed to embed query: {e}") from e

        search_query = SearchQuery(
            text=query,
            top_k=top_k * 2 if rerank and self._reranker else top_k,
            filters=filters or [],
            score_threshold=score_threshold,
        )

        try:
            results = await self._vector_store.search(search_query, query_vector)
        except Exception as e:
            raise RetrievalError(f"Vector search failed: {e}") from e

        if rerank and self._reranker and results:
            rerank_top_k = self._default_rerank_top_k
            results = await self._reranker.rerank(query, results, rerank_top_k)
        else:
            results = results[:top_k]

        return results
