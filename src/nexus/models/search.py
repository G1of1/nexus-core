"""Search and retrieval domain models."""

from typing import Any

from pydantic import BaseModel, Field


class SearchFilter(BaseModel):
    """Metadata filter for vector search."""

    field: str
    operator: str = "eq"  # eq, ne, in, gt, lt, gte, lte
    value: Any


class SearchQuery(BaseModel):
    """Parameters for a retrieval query."""

    text: str
    top_k: int = 5
    filters: list[SearchFilter] = Field(default_factory=list)
    score_threshold: float | None = None
    include_vectors: bool = False


class SearchResult(BaseModel):
    """A single result from vector similarity search."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", self.document_id))
