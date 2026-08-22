"""Shared test fixtures."""

import pytest

from nexus.config import NexusSettings
from nexus.pipeline.rag import RAGEngine
from nexus.providers.embeddings.mock import MockEmbeddingProvider
from nexus.providers.llm.mock import MockLLMProvider
from nexus.providers.vectorstore.memory import InMemoryVectorStore


@pytest.fixture
def settings() -> NexusSettings:
    return NexusSettings(
        chunk_size=100,
        chunk_overlap=20,
        top_k=3,
        rerank_top_k=2,
        vector_size=8,
    )


@pytest.fixture
def embedding_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dimension=8)


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def llm_provider() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def rag_engine(
    vector_store: InMemoryVectorStore,
    embedding_provider: MockEmbeddingProvider,
    llm_provider: MockLLMProvider,
    settings: NexusSettings,
) -> RAGEngine:
    return RAGEngine(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        settings=settings,
    )
