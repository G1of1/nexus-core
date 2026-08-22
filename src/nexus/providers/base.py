"""Abstract provider interfaces for dependency injection.

All infrastructure dependencies (vector stores, embedding models, LLMs) are
accessed through these protocols. Concrete implementations live in subpackages
and can be swapped without changing business logic.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from nexus.models.chunk import Chunk
from nexus.models.generation import GenerationRequest, GenerationResponse, StreamChunk
from nexus.models.search import SearchQuery, SearchResult


class EmbeddingProvider(ABC):
    """Generates vector embeddings for documents and queries."""

    @abstractmethod
    async def embed_document(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks (may use a different model or preprocessing)."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query (may use query-specific preprocessing)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""


class VectorStore(ABC):
    """Persists and searches vector embeddings."""

    @abstractmethod
    async def insert(self, chunks: list[Chunk]) -> None:
        """Insert chunk embeddings and metadata."""

    @abstractmethod
    async def search(self, query: SearchQuery, query_vector: list[float]) -> list[SearchResult]:
        """Perform similarity search with optional metadata filtering."""

    @abstractmethod
    async def delete(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""

    @abstractmethod
    async def delete_collection(self) -> None:
        """Delete the entire collection (use with caution)."""

    @abstractmethod
    async def ensure_collection(self) -> None:
        """Create collection if it does not exist."""


class LLMProvider(ABC):
    """Generates answers from retrieved context."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a complete answer."""

    @abstractmethod
    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Stream answer tokens as they are generated."""


class RerankerProvider(ABC):
    """Reranks retrieval results for improved relevance."""

    @abstractmethod
    async def rerank(
        self, query: str, results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Rerank search results and return top_k."""
