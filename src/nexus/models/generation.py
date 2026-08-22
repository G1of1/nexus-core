"""Generation domain models."""

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from nexus.models.search import SearchResult


class Citation(BaseModel):
    """Reference to a source document used in generation."""

    chunk_id: str
    document_id: str
    source: str
    excerpt: str
    score: float


class GenerationRequest(BaseModel):
    """Request to generate an answer from retrieved context."""

    query: str
    context: list[SearchResult] = Field(default_factory=list)
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    """Complete generated answer with citations."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamChunk(BaseModel):
    """A single token/chunk from a streaming generation response."""

    content: str
    is_final: bool = False
    citations: list[Citation] | None = None


StreamIterator = AsyncIterator[StreamChunk]
