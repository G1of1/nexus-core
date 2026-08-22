"""Google Gemini embedding provider."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from nexus.config import NexusSettings
from nexus.exceptions import ConfigurationError, EmbeddingError
from nexus.providers.base import EmbeddingProvider

# Known embedding dimensions for common Gemini models
_MODEL_DIMENSIONS: dict[str, int] = {
    "models/embedding-001": 768,
    "models/text-embedding-004": 768,
    "gemini-embedding-001": 768,
    "text-embedding-004": 768,
}


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using Google's Gemini embedding API."""

    def __init__(self, settings: NexusSettings) -> None:
        if not settings.gemini_api_key:
            raise ConfigurationError("NEXUS_GEMINI_API_KEY is required for Gemini embeddings")
        self._settings = settings
        self._client = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.gemini_api_key,
        )
        self._dimension = _MODEL_DIMENSIONS.get(settings.embedding_model, settings.vector_size)

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_document(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return await self._client.aembed_documents(texts)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed documents: {e}") from e

    async def embed_query(self, text: str) -> list[float]:
        try:
            return await self._client.aembed_query(text)
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}") from e
