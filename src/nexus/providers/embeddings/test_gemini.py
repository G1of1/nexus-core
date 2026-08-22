"""Tests for the GeminiEmbeddingProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.config import NexusSettings
from nexus.exceptions import ConfigurationError, EmbeddingError
from nexus.processing.loaders import DocumentLoader
from nexus.providers.embeddings.gemini import GeminiEmbeddingProvider


def test_init_requires_api_key():
    """Test that GeminiEmbeddingProvider raises ConfigurationError if API key is missing."""
    settings = NexusSettings(gemini_api_key=None)
    with pytest.raises(ConfigurationError, match="NEXUS_GEMINI_API_KEY is required"):
        GeminiEmbeddingProvider(settings)


@patch("nexus.providers.embeddings.gemini.GoogleGenerativeAIEmbeddings")
def test_init_success(mock_client):
    """Test successful initialization of GeminiEmbeddingProvider."""
    settings = NexusSettings(gemini_api_key="test-key", embedding_model="models/embedding-001")
    provider = GeminiEmbeddingProvider(settings)
    mock_client.assert_called_once_with(model="models/embedding-001", api_key="test-key")
    assert provider.dimension == 768


@patch("nexus.providers.embeddings.gemini.GoogleGenerativeAIEmbeddings")
def test_init_dimension_fallback(mock_client):
    """Test that dimension falls back to settings.vector_size for unknown models."""
    settings = NexusSettings(gemini_api_key="test-key", embedding_model="unknown-model", vector_size=128)
    provider = GeminiEmbeddingProvider(settings)
    assert provider.dimension == 128


@pytest.mark.asyncio
@patch("nexus.providers.embeddings.gemini.GoogleGenerativeAIEmbeddings")
async def test_embed_document(mock_client_cls):
    """Test embedding a list of documents."""
    mock_client_instance = MagicMock()
    mock_client_instance.aembed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    mock_client_cls.return_value = mock_client_instance

    settings = NexusSettings(gemini_api_key="test-key")
    provider = GeminiEmbeddingProvider(settings)

    texts = ["hello world", "foo bar"]
    embeddings = await provider.embed_document(texts)

    mock_client_instance.aembed_documents.assert_awaited_once_with(texts)
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
@patch("nexus.providers.embeddings.gemini.GoogleGenerativeAIEmbeddings")
async def test_embed_document_empty_list(mock_client_cls):
    """Test embedding an empty list of documents returns an empty list."""
    mock_client_instance = MagicMock()
    mock_client_instance.aembed_documents = AsyncMock()
    mock_client_cls.return_value = mock_client_instance

    settings = NexusSettings(gemini_api_key="test-key")
    provider = GeminiEmbeddingProvider(settings)

    embeddings = await provider.embed_document([])

    mock_client_instance.aembed_documents.assert_not_called()
    assert embeddings == []


@pytest.mark.asyncio
@patch("nexus.providers.embeddings.gemini.GoogleGenerativeAIEmbeddings")
async def test_embed_document_raises_embedding_error(mock_client_cls):
    """Test that exceptions during document embedding are wrapped in EmbeddingError."""
    mock_client_instance = MagicMock()
    mock_client_instance.aembed_documents = AsyncMock(side_effect=Exception("API error"))
    mock_client_cls.return_value = mock_client_instance

    settings = NexusSettings(gemini_api_key="test-key")
    provider = GeminiEmbeddingProvider(settings)

    with pytest.raises(EmbeddingError, match="Failed to embed documents: API error"):
        await provider.embed_document(["some text"])


@pytest.mark.asyncio
@patch("nexus.providers.embeddings.gemini.GoogleGenerativeAIEmbeddings")
async def test_embed_query(mock_client_cls):
    """Test embedding a single query."""
    mock_client_instance = MagicMock()
    mock_client_instance.aembed_query = AsyncMock(return_value=[0.5, 0.6])
    mock_client_cls.return_value = mock_client_instance

    settings = NexusSettings(gemini_api_key="test-key")
    provider = GeminiEmbeddingProvider(settings)

    text = "what is rag?"
    embedding = await provider.embed_query(text)

    mock_client_instance.aembed_query.assert_awaited_once_with(text)
    assert embedding == [0.5, 0.6]


@pytest.mark.asyncio
@patch("nexus.providers.embeddings.gemini.GoogleGenerativeAIEmbeddings")
async def test_embed_query_raises_embedding_error(mock_client_cls):
    """Test that exceptions during query embedding are wrapped in EmbeddingError."""
    mock_client_instance = MagicMock()
    mock_client_instance.aembed_query = AsyncMock(side_effect=Exception("API error"))
    mock_client_cls.return_value = mock_client_instance

    settings = NexusSettings(gemini_api_key="test-key")
    provider = GeminiEmbeddingProvider(settings)

    with pytest.raises(EmbeddingError, match="Failed to embed query: API error"):
        await provider.embed_query("some query")
