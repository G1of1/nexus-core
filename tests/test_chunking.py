"""Tests for document chunking."""

from nexus.config import NexusSettings
from nexus.processing.chunking import DocumentChunker, ChunkingStrategy
from nexus.processing.loaders import LoadedDocument


def test_chunk_text_splits_into_multiple_chunks():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
    text = "A" * 30 + " " + "B" * 30 + " " + "C" * 30
    chunks = chunker.chunk_text(text, document_id="doc-1")
    assert len(chunks) > 1
    assert all(c.document_id == "doc-1" for c in chunks)
    assert all(c.chunk_index == i for i, c in enumerate(chunks))


def test_chunk_document_preserves_metadata():
    settings = NexusSettings(chunk_size=100, chunk_overlap=10)
    chunker = DocumentChunker(settings=settings)
    loaded = LoadedDocument(
        filename="test.txt",
        content_type="text/plain",
        text="Hello world. " * 20,
        content_hash="abc123",
        metadata={"author": "test"},
    )
    chunks = chunker.chunk_document(loaded, document_id="doc-1")
    assert len(chunks) >= 1
    assert chunks[0].metadata["author"] == "test"
    assert chunks[0].metadata["filename"] == "test.txt"
    assert chunks[0].metadata["content_hash"] == "abc123"


def test_chunking_strategy_default():
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)
    assert chunker._strategy == ChunkingStrategy.RECURSIVE


def test_empty_text_produces_no_chunks():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_text("", document_id="doc-1")
    assert chunks == []


def test_chunk_text_respects_chunk_size_and_overlap():
    chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
    text = "".join(str(i % 10) for i in range(30))
    chunks = chunker.chunk_text(text, document_id="doc-2")
    assert len(chunks) > 1
    assert all(len(c.content) <= 10 for c in chunks)
    for i in range(1, len(chunks)):
        # consecutive chunks should overlap by the configured overlap
        assert chunks[i].content.startswith(chunks[i - 1].content[-2:])


def test_chunk_text_preserves_passed_metadata():
    chunker = DocumentChunker(chunk_size=15, chunk_overlap=5)
    text = "X" * 20
    meta = {"source": "unit-test"}
    chunks = chunker.chunk_text(text, document_id="doc-3", metadata=meta)
    assert all(c.metadata["source"] == "unit-test" for c in chunks)
    assert all(c.metadata["chunk_index"] == i for i, c in enumerate(chunks))


def test_single_chunk_when_text_smaller_than_chunk_size():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    text = "short text"
    chunks = chunker.chunk_text(text, document_id="doc-4")
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == text


def test_chunk_document_indices_sequential_and_unique():
    chunker = DocumentChunker(chunk_size=12, chunk_overlap=3)
    loaded = LoadedDocument(
        filename="seq.txt",
        content_type="text/plain",
        text="HELLO WORLD " * 5,
        content_hash="hashseq",
        metadata={},
    )
    chunks = chunker.chunk_document(loaded, document_id="doc-5")
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))
    assert len(set(indices)) == len(indices)


def test_strategy_can_be_overridden():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=5, strategy=ChunkingStrategy.SEMANTIC)
    assert chunker._strategy == ChunkingStrategy.SEMANTIC
