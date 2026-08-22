"""Mock embedding provider for testing."""

import hashlib

from nexus.providers.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embeddings based on text hash."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        values = [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]
        return values

    async def embed_document(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
