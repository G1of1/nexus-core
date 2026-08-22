"""Document loaders for supported formats (PDF, TXT, Markdown, DOCX)."""

import hashlib
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (Docx2txtLoader, PyPDFLoader,TextLoader)
from langchain_core.documents import Document as LCDocument
from pydantic import BaseModel, Field

from nexus.exceptions import DocumentLoadError, UnsupportedFormatError

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class LoadedDocument(BaseModel):
    """Raw document content extracted from a file."""

    filename: str
    content_type: str
    text: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    page_count: int | None = None


class DocumentLoader:
    """Loads and extracts text from supported document formats."""

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    @staticmethod
    def supported_extensions() -> list[str]:
        return list(SUPPORTED_EXTENSIONS.keys())

    @staticmethod
    def is_supported(path: str | Path) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def load(self, path: str | Path, metadata: dict[str, Any] | None = None) -> LoadedDocument:
        file_path = Path(path)
        if not file_path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported format '{suffix}'. Supported: {list(SUPPORTED_EXTENSIONS.keys())}"
            )

        try:
            lc_docs = self._load_with_langchain(file_path, suffix)
        except Exception as e:
            raise DocumentLoadError(f"Failed to load {file_path}: {e}") from e

        text = "\n\n".join(doc.page_content for doc in lc_docs if doc.page_content.strip())
        if not text.strip():
            raise DocumentLoadError(f"No text content extracted from {file_path}")

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        doc_metadata = metadata or {}
        doc_metadata.update({"source": str(file_path), "filename": file_path.name})

        return LoadedDocument(
            filename=file_path.name,
            content_type=SUPPORTED_EXTENSIONS[suffix],
            text=text,
            content_hash=content_hash,
            metadata=doc_metadata,
            page_count=len(lc_docs) if suffix == ".pdf" else None,
        )

    def load_bytes(
        self,
        content: bytes,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> LoadedDocument:
        """Load document from bytes by writing to a temp file."""
        import tempfile

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(f"Unsupported format '{suffix}'")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            return self.load(tmp_path, metadata={**(metadata or {}), "filename": filename})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _load_with_langchain(self, path: Path, suffix: str) -> list[LCDocument]:
        path_str = str(path)
        if suffix == ".pdf":
            return PyPDFLoader(path_str).load()
        if suffix in (".txt", ".md", ".markdown"):
            return TextLoader(path_str, encoding=self._encoding).load()
        if suffix == ".docx":
            return Docx2txtLoader(path_str).load()
        raise UnsupportedFormatError(f"No loader for {suffix}")
