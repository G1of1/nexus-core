"""Document domain models."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentVersion(BaseModel):
    """Tracks a specific version of an ingested document."""

    version_id: str = Field(default_factory=lambda: str(uuid4()))
    version_number: int = 1
    content_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    chunk_count: int = 0


class Document(BaseModel):
    """Represents a source document in the Nexus system."""

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    content_type: str
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)
    versions: list[DocumentVersion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: str | None = None

    @property
    def current_version(self) -> DocumentVersion | None:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.version_number)
