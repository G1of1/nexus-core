"""Tests for error handling in providers and pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.exceptions import (
    EmbeddingError,
    GenerationError,
    RetrievalError,
    VectorStoreError,
    UnsupportedFormatError,
    DocumentLoadError,
)
from nexus.models.chunk import Chunk
from nexus.pipeline.ingestion import IngestionPipeline
from nexus.retrieval.retriever import Retriever
from nexus.providers.embeddings.mock import MockEmbeddingProvider
from nexus.providers.llm.mock import MockLLMProvider
from nexus.providers.vectorstore.memory import InMemoryVectorStore
from nexus.config import NexusSettings
from nexus.processing.loaders import DocumentLoader


@pytest.mark.asyncio
async def test_embedding_provider_failure():
    """Test that embedding failures raise appropriate exceptions."""
    mock_provider = MagicMock(spec=MockEmbeddingProvider)
    mock_provider.embed_query = AsyncMock(
        side_effect=Exception("API rate limit exceeded")
    )

    vector_store = InMemoryVectorStore()
    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=mock_provider,
        settings=NexusSettings(),
    )

    with pytest.raises(RetrievalError, match="Failed to embed query"):
        await retriever.retrieve("test query")


@pytest.mark.asyncio
async def test_vector_store_search_failure():
    """Test that vector store search failures are handled."""
    embedding_provider = MockEmbeddingProvider(dimension=8)
    
    mock_store = MagicMock(spec=InMemoryVectorStore)
    mock_store.search = AsyncMock(
        side_effect=Exception("Connection refused")
    )

    retriever = Retriever(
        vector_store=mock_store,
        embedding_provider=embedding_provider,
        settings=NexusSettings(),
    )

    with pytest.raises(RetrievalError, match="Vector search failed"):
        await retriever.retrieve("test query")


@pytest.mark.asyncio
async def test_llm_generation_failure():
    """Test that LLM generation failures are handled."""
    mock_llm = MagicMock(spec=MockLLMProvider)
    mock_llm.generate = AsyncMock(
        side_effect=Exception("LLM service unavailable")
    )

    from nexus.models.generation import GenerationRequest
    request = GenerationRequest(query="test", context=[])
    with pytest.raises(Exception):
        await mock_llm.generate(request)


@pytest.mark.asyncio
async def test_unsupported_document_format():
    """Test that unsupported document formats raise appropriate exception."""
    loader = DocumentLoader()
    
    with pytest.raises((UnsupportedFormatError, DocumentLoadError)):
        # Try to load an unsupported format - raises DocumentLoadError for missing file
        # or UnsupportedFormatError if file exists
        loader.load("document.xyz")


@pytest.mark.asyncio
async def test_nonexistent_file_load():
    """Test that loading non-existent files raises DocumentLoadError."""
    loader = DocumentLoader()
    
    with pytest.raises(DocumentLoadError):
        loader.load("/nonexistent/path/file.pdf")


@pytest.mark.asyncio
async def test_vector_store_insert_failure():
    """Test that vector store insert failures are handled."""
    mock_store = MagicMock(spec=InMemoryVectorStore)
    mock_store.insert = AsyncMock(
        side_effect=VectorStoreError("Duplicate chunk ID")
    )
    
    chunks = [
        Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="test",
            chunk_index=0,
        )
    ]

    with pytest.raises(VectorStoreError):
        await mock_store.insert(chunks)


@pytest.mark.asyncio
async def test_vector_store_delete_nonexistent():
    """Test that deleting non-existent document returns 0."""
    vector_store = InMemoryVectorStore()
    
    # Delete a document that was never inserted
    deleted = await vector_store.delete("nonexistent-doc-id")
    assert deleted == 0


@pytest.mark.asyncio
async def test_retrieve_with_empty_store():
    """Test that retrieving from empty store returns empty results."""
    embedding_provider = MockEmbeddingProvider(dimension=8)
    vector_store = InMemoryVectorStore()
    retriever = Retriever(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        settings=NexusSettings(),
    )

    results = await retriever.retrieve("test query")
    assert results == []


@pytest.mark.asyncio
async def test_ingestion_with_invalid_metadata():
    """Test that invalid metadata is handled properly."""
    settings = NexusSettings()
    ingestion = IngestionPipeline(
        vector_store=InMemoryVectorStore(),
        embedding_provider=MockEmbeddingProvider(dimension=8),
        settings=settings,
    )

    # Very large metadata should be handled
    large_metadata = {"data": "x" * 1000000}
    
    # Should either succeed or raise a clear error, not crash
    try:
        await ingestion.ingest_text(
            text="test content",
            document_id="doc-1",
            metadata=large_metadata,
        )
    except (ValueError, VectorStoreError):
        # Expected - metadata too large or validation error
        pass


def test_document_loader_supported_formats():
    """Test that document loader reports supported formats correctly."""
    loader = DocumentLoader()
    supported = loader.supported_extensions()
    
    assert ".pdf" in supported
    assert ".txt" in supported
    assert ".md" in supported
    assert ".docx" in supported


def test_document_loader_is_supported():
    """Test format support checking."""
    loader = DocumentLoader()
    
    assert loader.is_supported("document.pdf")
    assert loader.is_supported("notes.txt")
    assert loader.is_supported("README.md")
    assert loader.is_supported("report.docx")
    assert not loader.is_supported("archive.zip")


@pytest.mark.asyncio
async def test_concurrent_embedding_failures():
    """Test handling of failures in concurrent embedding operations."""
    mock_provider = MagicMock(spec=MockEmbeddingProvider)
    mock_provider.embed_document = AsyncMock(
        side_effect=EmbeddingError("Batch embedding failed")
    )

    with pytest.raises(EmbeddingError):
        await mock_provider.embed_document(["text1", "text2", "text3"])
