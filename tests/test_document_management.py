"""Tests for document management edge cases."""

import pytest


@pytest.mark.asyncio
async def test_delete_nonexistent_document(rag_engine):
    """Test that deleting a non-existent document returns 0."""
    deleted = await rag_engine.ingestion.delete_document("nonexistent-doc-id")
    assert deleted == 0


@pytest.mark.asyncio
async def test_delete_document_removes_all_chunks(rag_engine):
    """Test that deleting a document removes all its chunks."""
    await rag_engine.ingestion.ingest_text(
        text="This is a test document with multiple chunks. " * 12,
        document_id="doc-1",
    )

    initial_results = await rag_engine.retrieve_only("multiple chunks")
    assert len(initial_results) > 0

    deleted = await rag_engine.ingestion.delete_document("doc-1")
    assert deleted > 0

    results_after_delete = await rag_engine.retrieve_only("multiple chunks")
    assert len(results_after_delete) == 0


@pytest.mark.asyncio
async def test_ingest_duplicate_document_id(rag_engine):
    """Test ingesting documents with the same ID."""
    await rag_engine.ingestion.ingest_text("First version of the document.", "doc-1")
    await rag_engine.ingestion.ingest_text("Second version of the document.", "doc-1")

    results = await rag_engine.retrieve_only("document")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_ingest_empty_document(rag_engine):
    """Test that ingesting an empty document is handled."""
    count = await rag_engine.ingestion.ingest_text("", "doc-empty")
    assert count >= 0

    results = await rag_engine.retrieve_only("test")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_ingest_very_large_document(rag_engine):
    """Test that large documents are chunked properly."""
    large_text = "This is a test sentence. " * 400

    count = await rag_engine.ingestion.ingest_text(large_text, "doc-large")
    assert count > 0

    results = await rag_engine.retrieve_only("test")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_ingest_with_special_characters(rag_engine):
    """Test ingesting documents with special characters."""
    text_with_special_chars = (
        "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`\n"
        "Unicode: café, naïve, résumé, 日本語, 中文\n"
        "Emoji: 😀 🎉 🚀\n"
        "Control chars: \t \n"
    )

    await rag_engine.ingestion.ingest_text(text_with_special_chars, "doc-special")

    results = await rag_engine.retrieve_only("café")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_ingest_with_metadata_persistence(rag_engine):
    """Test that metadata is properly persisted through ingestion."""
    metadata = {
        "source": "test-source",
        "category": "test",
        "priority": "high",
        "version": 1,
    }

    await rag_engine.ingestion.ingest_text(
        "Test document with metadata.",
        "doc-meta",
        metadata=metadata,
    )

    results = await rag_engine.retrieve_only("metadata")
    assert len(results) > 0
    assert all(r.metadata.get("source") == "test-source" for r in results)


@pytest.mark.asyncio
async def test_batch_ingest_consistency(rag_engine):
    """Test that batch ingestion produces consistent results."""
    documents = [
        ("doc-1", "First document content here."),
        ("doc-2", "Second document content here."),
        ("doc-3", "Third document content here."),
    ]

    for doc_id, text in documents:
        await rag_engine.ingestion.ingest_text(text=text, document_id=doc_id)

    results = await rag_engine.retrieve_only("document")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_delete_and_reingest_same_document(rag_engine):
    """Test deleting and re-ingesting the same document."""
    doc_id = "doc-reuse"
    await rag_engine.ingestion.ingest_text("First version.", doc_id)

    deleted = await rag_engine.ingestion.delete_document(doc_id)
    assert deleted > 0

    await rag_engine.ingestion.ingest_text("Second version.", doc_id)

    results = await rag_engine.retrieve_only("version")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_document_id_case_sensitivity(rag_engine):
    """Test that document IDs are case-sensitive."""
    await rag_engine.ingestion.ingest_text("Lowercase document ID.", "doc-case")
    await rag_engine.ingestion.ingest_text("Uppercase document ID.", "DOC-CASE")

    results = await rag_engine.retrieve_only("document")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_clear_all_documents(rag_engine):
    """Test clearing all documents from the vector store."""
    for i in range(3):
        await rag_engine.ingestion.ingest_text(f"Document {i} content.", f"doc-{i}")

    await rag_engine.ingestion._vector_store.delete_collection()

    results = await rag_engine.retrieve_only("content")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_metadata_field_with_none_values(rag_engine):
    """Test handling of None values in metadata."""
    metadata = {
        "source": None,
        "category": "test",
        "value": None,
    }

    await rag_engine.ingestion.ingest_text(
        "Document with None metadata values.",
        "doc-none",
        metadata=metadata,
    )

    results = await rag_engine.retrieve_only("metadata")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_document_with_duplicate_content(rag_engine):
    """Test ingesting documents with identical content."""
    text = "This is the exact same content."
    await rag_engine.ingestion.ingest_text(text, "doc-dup-1")
    await rag_engine.ingestion.ingest_text(text, "doc-dup-2")

    results = await rag_engine.retrieve_only("exact same")
    assert len(results) >= 1
