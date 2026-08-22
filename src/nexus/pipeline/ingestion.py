"""Document ingestion pipeline: load, chunk, embed, and index."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nexus.config import NexusSettings
from nexus.exceptions import DocumentLoadError, NexusError
from nexus.models.document import Document, DocumentStatus, DocumentVersion
from nexus.processing.chunking import DocumentChunker
from nexus.processing.loaders import DocumentLoader, LoadedDocument
from nexus.providers.base import EmbeddingProvider, VectorStore


class IngestionPipeline:
    """Processes documents through load → chunk → embed → store pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        settings: NexusSettings | None = None,
        loader: DocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._settings = settings
        self._loader = loader or DocumentLoader()
        self._chunker = chunker or DocumentChunker(settings)

    async def ingest_file(self, path: str | Path, metadata: dict[str, Any] | None = None, document: Document | None = None) -> Document:
        """Ingest a document from the filesystem."""
        file_path = Path(path)
        doc = document or Document(filename=file_path.name,content_type="application/octet-stream", metadata=metadata or {})
        doc.status = DocumentStatus.PROCESSING
        doc.updated_at = datetime.now(timezone.utc)

        try:
            loaded = self._loader.load(file_path, metadata={**(metadata or {}), **doc.metadata})
            return await self._ingest_loaded(loaded, doc)
        except DocumentLoadError:
            doc.status = DocumentStatus.FAILED
            raise
        except Exception as e:
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)
            raise NexusError(f"Ingestion failed for {file_path}: {e}") from e

    async def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        metadata: dict[str, Any] | None = None,
        document: Document | None = None,
    ) -> Document:
        """Ingest a document from raw bytes."""
        doc = document or Document(
            filename=filename,
            content_type="application/octet-stream",
            metadata=metadata or {},
        )
        doc.status = DocumentStatus.PROCESSING
        doc.updated_at = datetime.now(timezone.utc)

        try:
            loaded = self._loader.load_bytes(content, filename, metadata=metadata)
            return await self._ingest_loaded(loaded, doc)
        except Exception as e:
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)
            raise NexusError(f"Ingestion failed for {filename}: {e}") from e

    async def ingest_text(
        self,
        text: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Ingest raw text directly. Returns number of chunks indexed."""
        import hashlib

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        chunks = self._chunker.chunk_text(
            text,
            document_id=document_id,
            metadata={**(metadata or {}), "content_hash": content_hash},
        )
        return await self._embed_and_store(chunks)

    async def reindex_document(self, document_id: str, loaded: LoadedDocument) -> Document:
        """Re-index an existing document (creates new version)."""
        await self._vector_store.delete(document_id)
        doc = Document(
            document_id=document_id,
            filename=loaded.filename,
            content_type=loaded.content_type,
        )
        return await self._ingest_loaded(loaded, doc)

    async def delete_document(self, document_id: str) -> int:
        """Remove all chunks for a document from the vector store."""
        return await self._vector_store.delete(document_id)

    async def _ingest_loaded(self, loaded: LoadedDocument, doc: Document) -> Document:
        doc.filename = loaded.filename
        doc.content_type = loaded.content_type
        doc.metadata.update(loaded.metadata)

        # Check for existing version
        existing_version = doc.current_version
        version_number = (existing_version.version_number + 1) if existing_version else 1

        chunks = self._chunker.chunk_document(loaded, doc.document_id)
        chunk_count = await self._embed_and_store(chunks)

        version = DocumentVersion(
            version_number=version_number,
            content_hash=loaded.content_hash,
            chunk_count=chunk_count,
        )
        doc.versions.append(version)
        doc.status = DocumentStatus.INDEXED
        doc.updated_at = datetime.now(timezone.utc)
        doc.error_message = None
        return doc

    async def _embed_and_store(self, chunks: list) -> int:
        if not chunks:
            return 0
        texts = [c.content for c in chunks]
        embeddings = await self._embedding_provider.embed_document(texts)
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding
        await self._vector_store.insert(chunks)
        return len(chunks)
