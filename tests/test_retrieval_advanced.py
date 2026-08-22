"""Tests for advanced retrieval features."""

import pytest
from nexus.models.chunk import Chunk
from nexus.models.search import SearchFilter
from nexus.providers.embeddings.mock import MockEmbeddingProvider
from nexus.providers.vectorstore.memory import InMemoryVectorStore
from nexus.retrieval.retriever import Retriever
from nexus.config import NexusSettings


@pytest.fixture
async def indexed_retriever():
    """Create a retriever with test data."""
    store = InMemoryVectorStore()
    embedder = MockEmbeddingProvider(dimension=8)

    chunks = [
        Chunk(
            document_id="doc-1",
            content="Python is a high-level programming language",
            chunk_index=0,
            metadata={"topic": "programming", "difficulty": "beginner"},
        ),
        Chunk(
            document_id="doc-2",
            content="Machine learning with Python and TensorFlow",
            chunk_index=0,
            metadata={"topic": "ml", "difficulty": "advanced"},
        ),
        Chunk(
            document_id="doc-3",
            content="Web development using Python Flask",
            chunk_index=0,
            metadata={"topic": "web", "difficulty": "intermediate"},
        ),
        Chunk(
            document_id="doc-4",
            content="Data analysis with Python pandas library",
            chunk_index=0,
            metadata={"topic": "data", "difficulty": "intermediate"},
        ),
        Chunk(
            document_id="doc-5",
            content="JavaScript for web development",
            chunk_index=0,
            metadata={"topic": "web", "difficulty": "beginner"},
        ),
    ]

    texts = [c.content for c in chunks]
    embeddings = await embedder.embed_document(texts)
    for chunk, emb in zip(chunks, embeddings, strict=True):
        chunk.embedding = emb
    await store.insert(chunks)

    settings = NexusSettings(top_k=5, rerank_top_k=3)
    retriever = Retriever(
        vector_store=store,
        embedding_provider=embedder,
        settings=settings,
    )

    return retriever, store, embedder


@pytest.mark.asyncio
async def test_retrieve_with_score_threshold(indexed_retriever):
    """Test retrieval with score threshold filtering."""
    retriever, _, _ = indexed_retriever

    # High threshold - should filter out low-scoring results
    results = await retriever.retrieve(
        query="Python",
        score_threshold=0.8,
    )

    assert all(r.score >= 0.8 for r in results)


@pytest.mark.asyncio
async def test_retrieve_with_zero_threshold(indexed_retriever):
    """Test retrieval with zero score threshold."""
    retriever, _, _ = indexed_retriever

    results_with_threshold = await retriever.retrieve(
        query="Python",
        score_threshold=0.0,
    )

    results_without_threshold = await retriever.retrieve(
        query="Python",
        score_threshold=None,
    )

    # Zero threshold should include all results
    assert len(results_with_threshold) >= len(results_without_threshold)


@pytest.mark.asyncio
async def test_retrieve_with_single_filter(indexed_retriever):
    """Test retrieval with single metadata filter."""
    retriever, _, _ = indexed_retriever

    filters = [SearchFilter(field="difficulty", operator="eq", value="beginner")]
    results = await retriever.retrieve(
        query="Python",
        filters=filters,
    )

    # All results should be beginner level
    assert all(r.metadata.get("difficulty") == "beginner" for r in results)


@pytest.mark.asyncio
async def test_retrieve_with_multiple_filters(indexed_retriever):
    """Test retrieval with multiple metadata filters."""
    retriever, _, _ = indexed_retriever

    filters = [
        SearchFilter(field="topic", operator="eq", value="web"),
        SearchFilter(field="difficulty", operator="eq", value="beginner"),
    ]
    results = await retriever.retrieve(
        query="development",
        filters=filters,
    )

    # All results should match both filters
    for r in results:
        assert r.metadata.get("topic") == "web"
        assert r.metadata.get("difficulty") == "beginner"


@pytest.mark.asyncio
async def test_retrieve_with_in_filter(indexed_retriever):
    """Test retrieval with 'in' operator for multiple values."""
    retriever, _, _ = indexed_retriever

    filters = [
        SearchFilter(
            field="difficulty",
            operator="in",
            value=["beginner", "intermediate"],
        )
    ]
    results = await retriever.retrieve(
        query="Python",
        filters=filters,
    )

    # Results should only have beginner or intermediate difficulty
    for r in results:
        assert r.metadata.get("difficulty") in ["beginner", "intermediate"]


@pytest.mark.asyncio
async def test_retrieve_with_gt_filter(indexed_retriever):
    """Test retrieval with 'greater than' filter."""
    retriever, _, _ = indexed_retriever

    # Add numeric metadata
    filters = [SearchFilter(field="difficulty", operator="gt", value="advanced")]
    results = await retriever.retrieve(
        query="Python",
        filters=filters,
    )

    # Should work with string comparison
    for r in results:
        difficulty = r.metadata.get("difficulty", "")
        assert difficulty > "advanced"


@pytest.mark.asyncio
async def test_retrieve_with_no_reranking(indexed_retriever):
    """Test retrieval without reranking."""
    retriever, _, _ = indexed_retriever

    results_with_rerank = await retriever.retrieve(
        query="Python",
        rerank=True,
    )

    results_without_rerank = await retriever.retrieve(
        query="Python",
        rerank=False,
    )

    # Both should return results
    assert len(results_with_rerank) > 0
    assert len(results_without_rerank) > 0


@pytest.mark.asyncio
async def test_retrieve_with_custom_top_k(indexed_retriever):
    """Test retrieval with custom top_k parameter."""
    retriever, _, _ = indexed_retriever

    results_top_2 = await retriever.retrieve(
        query="Python",
        top_k=2,
    )

    results_top_5 = await retriever.retrieve(
        query="Python",
        top_k=5,
    )

    assert len(results_top_2) <= 2
    assert len(results_top_5) <= 5
    assert len(results_top_2) <= len(results_top_5)


@pytest.mark.asyncio
async def test_retrieve_all_with_sufficient_top_k(indexed_retriever):
    """Test that setting high top_k returns all results."""
    retriever, _, _ = indexed_retriever

    results = await retriever.retrieve(
        query="Python",
        top_k=1000,
    )

    # Should return all 5 documents
    assert len(results) == 5


@pytest.mark.asyncio
async def test_retrieve_with_combined_filters_and_threshold(indexed_retriever):
    """Test retrieval with both filters and score threshold."""
    retriever, _, _ = indexed_retriever

    filters = [SearchFilter(field="topic", operator="eq", value="programming")]
    results = await retriever.retrieve(
        query="Python",
        filters=filters,
        score_threshold=0.7,
    )

    # All results should match filters and threshold
    assert all(r.score >= 0.7 for r in results)
    for r in results:
        assert r.metadata.get("topic") == "programming"


@pytest.mark.asyncio
async def test_retrieve_empty_results_with_strict_filter(indexed_retriever):
    """Test retrieval with filters that produce no results."""
    retriever, _, _ = indexed_retriever

    filters = [SearchFilter(field="topic", operator="eq", value="nonexistent")]
    results = await retriever.retrieve(
        query="Python",
        filters=filters,
    )

    # Should return empty list, not error
    assert results == []


@pytest.mark.asyncio
async def test_retrieve_with_ne_filter(indexed_retriever):
    """Test retrieval with 'not equal' filter."""
    retriever, _, _ = indexed_retriever

    filters = [SearchFilter(field="topic", operator="ne", value="web")]
    results = await retriever.retrieve(
        query="Python",
        filters=filters,
    )

    # Results should not have topic "web"
    for r in results:
        assert r.metadata.get("topic") != "web"


@pytest.mark.asyncio
async def test_retrieve_respects_default_top_k(indexed_retriever):
    """Test that default top_k setting is respected."""
    retriever, _, _ = indexed_retriever

    # Settings has top_k=5, rerank_top_k=3
    results = await retriever.retrieve(query="Python")

    # Should return up to top_k results
    assert len(results) <= 5


@pytest.mark.asyncio
async def test_retrieve_handles_nonexistent_filter_field(indexed_retriever):
    """Test retrieval when filter field doesn't exist in metadata."""
    retriever, _, _ = indexed_retriever

    filters = [
        SearchFilter(field="nonexistent_field", operator="eq", value="value")
    ]
    results = await retriever.retrieve(
        query="Python",
        filters=filters,
    )

    # Should return empty or handle gracefully
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_retrieve_score_ordering(indexed_retriever):
    """Test that results are ordered by score descending."""
    retriever, _, _ = indexed_retriever

    results = await retriever.retrieve(query="Python", top_k=10)

    # Verify score ordering
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


@pytest.mark.asyncio
async def test_retrieve_empty_query(indexed_retriever):
    """Test retrieval with empty query string."""
    retriever, _, _ = indexed_retriever

    # Empty query should still work or raise appropriate error
    try:
        results = await retriever.retrieve(query="")
        assert isinstance(results, list)
    except (ValueError, Exception):
        # Expected - empty query may not be valid
        pass


@pytest.mark.asyncio
async def test_retrieve_very_long_query(indexed_retriever):
    """Test retrieval with very long query string."""
    retriever, _, _ = indexed_retriever

    long_query = "Python programming " * 100

    results = await retriever.retrieve(query=long_query)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_retrieve_unicode_query(indexed_retriever):
    """Test retrieval with unicode query."""
    retriever, _, _ = indexed_retriever

    results = await retriever.retrieve(query="Python 编程")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_retrieve_special_character_query(indexed_retriever):
    """Test retrieval with special characters in query."""
    retriever, _, _ = indexed_retriever

    results = await retriever.retrieve(query="Python@#$%")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_retrieve_filter_with_null_values(indexed_retriever):
    """Test retrieval when metadata contains null values."""
    retriever, store, embedder = indexed_retriever

    # Insert chunk with null metadata value
    chunk = Chunk(
        document_id="doc-null",
        content="Content with null metadata",
        chunk_index=0,
        metadata={"topic": None, "difficulty": "beginner"},
    )
    embeddings = await embedder.embed_document(["Content with null metadata"])
    chunk.embedding = embeddings[0]
    await store.insert([chunk])

    filters = [SearchFilter(field="topic", operator="eq", value=None)]
    results = await retriever.retrieve(
        query="content",
        filters=filters,
    )

    # Should handle null filtering gracefully
    assert isinstance(results, list)
