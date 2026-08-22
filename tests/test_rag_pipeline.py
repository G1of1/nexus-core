"""Tests for end-to-end RAG pipeline."""

import pytest

from nexus.models.search import SearchFilter


async def test_ingest_text_and_query(rag_engine):
    await rag_engine.ingestion.ingest_text(
        text="The Nexus RAG platform supports PDF, TXT, Markdown, and DOCX formats. "
        "It uses vector search for retrieval and LLMs for generation.",
        document_id="doc-nexus",
        metadata={"source": "docs"},
    )

    response = await rag_engine.query("What document formats does Nexus support?")
    assert response.answer
    assert "Mock answer" in response.answer
    assert len(response.retrieved_chunks) > 0


async def test_retrieve_only(rag_engine):
    await rag_engine.ingestion.ingest_text(
        text="Redis is used for caching in the Nexus platform.",
        document_id="doc-redis",
    )
    results = await rag_engine.retrieve_only("caching")
    assert isinstance(results, list)


async def test_query_stream(rag_engine):
    await rag_engine.ingestion.ingest_text(
        text="Qdrant is the vector database used by Nexus.",
        document_id="doc-qdrant",
    )
    chunks = []
    async for chunk in rag_engine.query_stream("What vector database does Nexus use?"):
        chunks.append(chunk)
    assert len(chunks) > 0
    assert chunks[-1].is_final


async def test_metadata_filtering_in_query(rag_engine):
    await rag_engine.ingestion.ingest_text(
        text="Public information about APIs.",
        document_id="doc-public",
        metadata={"access": "public"},
    )
    await rag_engine.ingestion.ingest_text(
        text="Private internal documentation.",
        document_id="doc-private",
        metadata={"access": "private"},
    )

    filters = [SearchFilter(field="access", operator="eq", value="public")]
    results = await rag_engine.retrieve_only("APIs", filters=filters)
    assert all(r.metadata.get("access") == "public" for r in results)


async def test_delete_document(rag_engine):
    await rag_engine.ingestion.ingest_text(
        text="Temporary document content.",
        document_id="doc-temp",
    )
    deleted = await rag_engine.ingestion.delete_document("doc-temp")
    assert deleted >= 0
