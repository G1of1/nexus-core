"""Tests for reranking implementation."""

import pytest
from nexus.models.search import SearchResult
from nexus.retrieval.reranker import ScoreReranker
from nexus.config import NexusSettings


@pytest.fixture
def reranker():
    """Create a reranker for testing."""
    return ScoreReranker()


@pytest.fixture
def sample_results():
    """Create sample search results for testing."""
    return [
        SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Python programming language",
            score=0.85,
            metadata={"source": "doc-1"},
        ),
        SearchResult(
            chunk_id="chunk-2",
            document_id="doc-2",
            content="Java programming language",
            score=0.82,
            metadata={"source": "doc-2"},
        ),
        SearchResult(
            chunk_id="chunk-3",
            document_id="doc-3",
            content="Programming best practices",
            score=0.78,
            metadata={"source": "doc-3"},
        ),
        SearchResult(
            chunk_id="chunk-4",
            document_id="doc-4",
            content="C++ programming",
            score=0.75,
            metadata={"source": "doc-4"},
        ),
        SearchResult(
            chunk_id="chunk-5",
            document_id="doc-5",
            content="Web development tutorials",
            score=0.65,
            metadata={"source": "doc-5"},
        ),
    ]


@pytest.mark.asyncio
async def test_reranker_returns_top_k(reranker, sample_results):
    """Test that reranker returns exactly top_k results."""
    reranked = await reranker.rerank(
        query="Python programming",
        results=sample_results,
        top_k=3,
    )
    assert len(reranked) == 3


@pytest.mark.asyncio
async def test_reranker_respects_top_k_less_than_results(reranker, sample_results):
    """Test reranker when top_k is less than available results."""
    reranked = await reranker.rerank(
        query="programming",
        results=sample_results,
        top_k=2,
    )
    assert len(reranked) == 2


@pytest.mark.asyncio
async def test_reranker_handles_top_k_exceeds_results(reranker, sample_results):
    """Test reranker when top_k exceeds number of results."""
    reranked = await reranker.rerank(
        query="programming",
        results=sample_results,
        top_k=10,
    )
    assert len(reranked) == len(sample_results)


@pytest.mark.asyncio
async def test_reranker_maintains_relevance_order(reranker, sample_results):
    """Test that reranker maintains or improves result ordering."""
    reranked = await reranker.rerank(
        query="Python programming",
        results=sample_results,
        top_k=3,
    )

    # Results should still be in descending order by score
    for i in range(len(reranked) - 1):
        assert reranked[i].score >= reranked[i + 1].score


@pytest.mark.asyncio
async def test_reranker_with_empty_results(reranker):
    """Test reranker with empty results list."""
    reranked = await reranker.rerank(
        query="test",
        results=[],
        top_k=3,
    )
    assert reranked == []


@pytest.mark.asyncio
async def test_reranker_with_single_result(reranker):
    """Test reranker with single result."""
    single_result = SearchResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Test content",
        score=0.9,
        metadata={"source": "doc-1"},
    )

    reranked = await reranker.rerank(
        query="test",
        results=[single_result],
        top_k=1,
    )
    assert len(reranked) == 1
    assert reranked[0].chunk_id == "chunk-1"


@pytest.mark.asyncio
async def test_reranker_top_k_zero(reranker, sample_results):
    """Test reranker behavior with top_k=0."""
    reranked = await reranker.rerank(
        query="programming",
        results=sample_results,
        top_k=0,
    )
    assert len(reranked) == 0


@pytest.mark.asyncio
async def test_reranker_top_k_negative(reranker, sample_results):
    """Test reranker behavior with negative top_k."""
    results = await reranker.rerank(
        query="programming",
        results=sample_results,
        top_k=-1,
    )
    assert results == []


@pytest.mark.asyncio
async def test_reranker_preserves_metadata(reranker, sample_results):
    """Test that reranker preserves result metadata."""
    reranked = await reranker.rerank(
        query="programming",
        results=sample_results,
        top_k=3,
    )

    for result in reranked:
        assert result.metadata is not None
        assert "source" in result.metadata


@pytest.mark.asyncio
async def test_reranker_score_reranking_prioritizes_relevance(reranker, sample_results):
    """Test that score reranker respects original scores."""
    # ScoreReranker should use the provided scores
    reranked = await reranker.rerank(
        query="programming",
        results=sample_results,
        top_k=3,
    )

    # The top result should be the one with highest score
    assert reranked[0].score >= reranked[1].score


@pytest.mark.asyncio
async def test_reranker_with_identical_scores(reranker):
    """Test reranker behavior when multiple results have identical scores."""
    results = [
        SearchResult(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            content=f"Content {i}",
            score=0.8,
            metadata={"source": f"doc-{i}"},
        )
        for i in range(5)
    ]

    reranked = await reranker.rerank(
        query="test",
        results=results,
        top_k=3,
    )
    assert len(reranked) == 3


@pytest.mark.asyncio
async def test_reranker_long_query(reranker, sample_results):
    """Test reranker with very long query string."""
    long_query = "Python programming " * 100  # Very long query

    reranked = await reranker.rerank(
        query=long_query,
        results=sample_results,
        top_k=3,
    )
    assert len(reranked) == 3


@pytest.mark.asyncio
async def test_reranker_special_character_query(reranker, sample_results):
    """Test reranker with special characters in query."""
    special_query = "Python@#$% & programming!"

    reranked = await reranker.rerank(
        query=special_query,
        results=sample_results,
        top_k=3,
    )
    assert len(reranked) == 3


@pytest.mark.asyncio
async def test_score_reranker_implementation(reranker, sample_results):
    """Test that ScoreReranker sorts by score correctly."""
    # Shuffle the results
    shuffled = sample_results[::-1]  # Reverse order

    reranked = await reranker.rerank(
        query="programming",
        results=shuffled,
        top_k=len(shuffled),
    )

    # Should be sorted by score descending
    for i in range(len(reranked) - 1):
        assert reranked[i].score >= reranked[i + 1].score


@pytest.mark.asyncio
async def test_reranker_with_floating_point_scores(reranker):
    """Test reranker with various floating point score values."""
    results = [
        SearchResult(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            content=f"Content {i}",
            score=float(i) / 10,  # 0.0, 0.1, 0.2, etc.
            metadata={"source": f"doc-{i}"},
        )
        for i in range(1, 6)
    ]

    reranked = await reranker.rerank(
        query="test",
        results=results,
        top_k=3,
    )

    # Should be in descending order
    assert reranked[0].score > reranked[1].score
    assert reranked[1].score > reranked[2].score


@pytest.mark.asyncio
async def test_reranker_unicode_content(reranker):
    """Test reranker with unicode content in results."""
    results = [
        SearchResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Python 编程语言",
            score=0.9,
            metadata={"source": "doc-1"},
        ),
        SearchResult(
            chunk_id="chunk-2",
            document_id="doc-2",
            content="Java プログラミング言語",
            score=0.85,
            metadata={"source": "doc-2"},
        ),
        SearchResult(
            chunk_id="chunk-3",
            document_id="doc-3",
            content="برمجة C++",
            score=0.8,
            metadata={"source": "doc-3"},
        ),
    ]

    reranked = await reranker.rerank(
        query="programming",
        results=results,
        top_k=3,
    )
    assert len(reranked) == 3
    assert all(r.content for r in reranked)
