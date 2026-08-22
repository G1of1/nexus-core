"""Environment-based configuration for Nexus core."""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()


class NexusSettings(BaseSettings):
    """All Nexus settings loaded from environment variables with NEXUS_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="NEXUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Vector store
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = ""
    vector_size: int = 768

    # Gemini
    gemini_api_key: SecretStr | None = None
    embedding_model: str = "models/text-embedding-004"
    embedding_batch_size: int = 100

    # LLM
    llm_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5
    rerank_top_k: int = 3
    score_threshold: float | None = None
    enable_hybrid_search: bool = False

    # Observability
    langsmith_api_key: str | None = None
    langsmith_project: str = "nexus"
    enable_tracing: bool = Field(default=False)

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def chunk_overlap_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("chunk_overlap must be non-negative")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def chunk_overlap_less_than_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_k must be positive")
        return v

    @field_validator("rerank_top_k")
    @classmethod
    def rerank_top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("rerank_top_k must be positive")
        return v

    @field_validator("rerank_top_k")
    @classmethod
    def rerank_top_k_not_exceeds_top_k(cls, v: int, info) -> int:
        top_k = info.data.get("top_k", 5)
        if v > top_k:
            raise ValueError("rerank_top_k must not exceed top_k")
        return v

    @field_validator("vector_size")
    @classmethod
    def vector_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("vector_size must be positive")
        return v

    @field_validator("llm_temperature")
    @classmethod
    def temperature_in_range(cls, v: float) -> float:
        if not (0 <= v <= 2):
            raise ValueError("llm_temperature must be between 0 and 2")
        return v

    @field_validator("llm_max_tokens")
    @classmethod
    def max_tokens_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("llm_max_tokens must be positive")
        return v

    @field_validator("embedding_batch_size")
    @classmethod
    def embedding_batch_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("embedding_batch_size must be positive")
        return v

    @field_validator("score_threshold")
    @classmethod
    def score_threshold_in_range(cls, v: float | None) -> float | None:
        if v is not None and not (0 <= v <= 1):
            raise ValueError("score_threshold must be between 0 and 1 or None")
        return v
