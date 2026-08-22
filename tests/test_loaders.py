"""Tests for document loaders."""

import tempfile
from pathlib import Path

import pytest

from nexus.exceptions import DocumentLoadError, UnsupportedFormatError
from nexus.processing.loaders import DocumentLoader


def test_supported_extensions():
    exts = DocumentLoader.supported_extensions()
    assert ".pdf" in exts
    assert ".txt" in exts
    assert ".md" in exts
    assert ".docx" in exts


def test_is_supported():
    assert DocumentLoader.is_supported("report.pdf")
    assert DocumentLoader.is_supported("notes.txt")
    assert not DocumentLoader.is_supported("image.png")


def test_load_txt_file():
    loader = DocumentLoader()
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
        f.write("Hello, Nexus RAG platform!")
        tmp_path = f.name

    try:
        doc = loader.load(tmp_path)
        assert "Nexus RAG" in doc.text
        assert doc.content_type == "text/plain"
        assert len(doc.content_hash) == 64
    finally:
        Path(tmp_path).unlink()


def test_load_markdown_file():
    loader = DocumentLoader()
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
        f.write("# Title\n\nSome markdown content.")
        tmp_path = f.name

    try:
        doc = loader.load(tmp_path)
        assert "markdown content" in doc.text
    finally:
        Path(tmp_path).unlink()


def test_load_nonexistent_file():
    loader = DocumentLoader()
    with pytest.raises(DocumentLoadError, match="File not found"):
        loader.load("/nonexistent/file.txt")


def test_load_unsupported_format():
    loader = DocumentLoader()
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        tmp_path = f.name
    try:
        with pytest.raises(UnsupportedFormatError):
            loader.load(tmp_path)
    finally:
        Path(tmp_path).unlink()


def test_load_bytes():
    loader = DocumentLoader()
    content = b"Content loaded from bytes."
    doc = loader.load_bytes(content, "test.txt")
    assert "Content loaded from bytes" in doc.text
