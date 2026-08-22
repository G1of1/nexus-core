"""Document chunking strategies."""

from enum import StrEnum

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from nexus.config import NexusSettings
from nexus.models.chunk import Chunk
from nexus.processing.loaders import LoadedDocument


class ChunkingStrategy(StrEnum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"  # Reserved for future semantic chunking


class DocumentChunker:
    """Splits documents into semantic chunks for embedding."""

    def __init__(self, settings: NexusSettings | None = None, chunk_size: int | None = None, chunk_overlap: int | None = None, strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    ) -> None:
        self._settings = settings
        self._chunk_size = chunk_size or (settings.chunk_size if settings else 512)
        self._chunk_overlap = chunk_overlap or (settings.chunk_overlap if settings else 64)
        self._strategy = strategy
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, loaded: LoadedDocument, document_id: str) -> list[Chunk]:
        texts = self._splitter.split_text(loaded.text)
        chunks: list[Chunk] = []
        for index, text in enumerate(texts):
            chunk_metadata = {
                **loaded.metadata,
                "chunk_index": index,
                "content_hash": loaded.content_hash,
                "filename": loaded.filename,
            }
            chunks.append(
                Chunk(
                    document_id=document_id,
                    content=text,
                    chunk_index=index,
                    metadata=chunk_metadata,
                )
            )
        return chunks

    def chunk_text(self, text: str, document_id: str, metadata: dict | None = None) -> list[Chunk]:
        texts = self._splitter.split_text(text)
        base_metadata = metadata or {}
        return [
            Chunk(
                document_id=document_id,
                content=t,
                chunk_index=i,
                metadata={**base_metadata, "chunk_index": i},
            )
            for i, t in enumerate(texts)
        ]


class ChunkConfig(BaseModel):
    """Configuration for chunking behavior."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
