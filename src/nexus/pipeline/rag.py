"""End-to-end RAG engine orchestrating ingestion, retrieval, and generation."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nexus.config import NexusSettings
from nexus.models.generation import Citation, GenerationRequest, StreamChunk
from nexus.models.search import SearchFilter, SearchResult
from nexus.pipeline.ingestion import IngestionPipeline
from nexus.providers.base import EmbeddingProvider, LLMProvider, RerankerProvider, VectorStore
from nexus.retrieval.reranker import ScoreReranker
from nexus.retrieval.retriever import Retriever


class RAGResponse(BaseModel):
    """Complete RAG query response with answer and retrieval metadata."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[SearchResult] = Field(default_factory=list)
    model: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)


class RAGEngine:
    """Main entry point for RAG operations.

    Combines ingestion, retrieval, and generation into a unified interface.
    All infrastructure is injected via provider interfaces.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        settings: NexusSettings | None = None,
        reranker: RerankerProvider | None = None,
    ) -> None:
        self._settings = settings
        self._llm_provider = llm_provider
        self._reranker = reranker or ScoreReranker()

        self._retriever = Retriever(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            settings=settings,
            reranker=self._reranker,
        )
        self._ingestion = IngestionPipeline(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            settings=settings,
        )

    @property
    def ingestion(self) -> IngestionPipeline:
        return self._ingestion

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    async def ingest_file(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ):
        """Ingest a document from the filesystem."""
        return await self._ingestion.ingest_file(path, metadata=metadata)

    async def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ):
        """Ingest a document from raw bytes."""
        return await self._ingestion.ingest_bytes(content, filename, metadata=metadata)

    async def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: list[SearchFilter] | None = None,
        system_prompt: str | None = None,
        rerank: bool = True,
    ) -> RAGResponse:
        """Retrieve context and generate an answer."""
        context = await self._retriever.retrieve(
            query=question,
            top_k=top_k,
            filters=filters,
            rerank=rerank,
        )

        request = GenerationRequest(
            query=question,
            context=context,
            system_prompt=system_prompt,
        )
        response = await self._llm_provider.generate(request)

        return RAGResponse(
            answer=response.answer,
            citations=response.citations,
            retrieved_chunks=context,
            model=response.model,
            token_usage=response.token_usage,
        )

    async def query_stream(
        self,
        question: str,
        top_k: int | None = None,
        filters: list[SearchFilter] | None = None,
        system_prompt: str | None = None,
        rerank: bool = True,
    ) -> AsyncIterator[StreamChunk]:
        """Stream an answer with retrieved context."""
        context = await self._retriever.retrieve(
            query=question,
            top_k=top_k,
            filters=filters,
            rerank=rerank,
        )

        request = GenerationRequest(
            query=question,
            context=context,
            system_prompt=system_prompt,
        )
        async for chunk in self._llm_provider.stream(request):
            yield chunk

    async def retrieve_only(
        self,
        query: str,
        top_k: int | None = None,
        filters: list[SearchFilter] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks without generation."""
        return await self._retriever.retrieve(query=query, top_k=top_k, filters=filters)
