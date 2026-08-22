"""Tests for retrieval logic."""

import pytest

from nexus.models.chunk import Chunk
from nexus.models.search import SearchFilter
from nexus.providers.embeddings.mock import MockEmbeddingProvider
from nexus.providers.vectorstore.memory import InMemoryVectorStore
from nexus.retrieval.retriever import Retriever
from nexus.config import NexusSettings


@pytest.fixture
async def indexed_store():
    store = InMemoryVectorStore()
    embedder = MockEmbeddingProvider(dimension=8)

    chunks = [
        Chunk(
            document_id="doc-1",
            content="Python is a programming language.",
            chunk_index=0,
            metadata={"topic": "programming"},
        ),
        Chunk(
            document_id="doc-2",
            content="The Eiffel Tower is in Paris.",
            chunk_index=0,
            metadata={"topic": "travel"},
        ),
        Chunk(
            document_id="doc-3",
            content="Machine learning uses Python extensively.",
            chunk_index=0,
            metadata={"topic": "programming"},
        ),
    ]

    texts = [c.content for c in chunks]
    embeddings = await embedder.embed_document(texts)
    for chunk, emb in zip(chunks, embeddings, strict=True):
        chunk.embedding = emb
    await store.insert(chunks)
    return store, embedder


async def test_retrieve_returns_relevant_results(indexed_store):
    store, embedder = indexed_store
    retriever = Retriever(
        vector_store=store,
        embedding_provider=embedder,
        settings=NexusSettings(top_k=2, rerank_top_k=1),
    )
    results = await retriever.retrieve("Python programming")
    assert len(results) <= 2
    assert all(r.score > 0 for r in results)


async def test_retrieve_with_metadata_filter(indexed_store):
    store, embedder = indexed_store
    retriever = Retriever(
        vector_store=store,
        embedding_provider=embedder,
        settings=NexusSettings(top_k=5),
    )
    filters = [SearchFilter(field="topic", operator="eq", value="programming")]
    results = await retriever.retrieve("Python", filters=filters)
    assert all(r.metadata.get("topic") == "programming" for r in results)


async def test_retrieve_empty_store():
    store = InMemoryVectorStore()
    embedder = MockEmbeddingProvider()
    retriever = Retriever(vector_store=store, embedding_provider=embedder)
    results = await retriever.retrieve("anything")
    assert results == []
