"""Chunk domain models."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A semantic chunk derived from a source document."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", self.document_id))
