"""Domain exceptions for Nexus core."""


class NexusError(Exception):
    """Base exception for all Nexus errors."""


class DocumentLoadError(NexusError):
    """Failed to load or parse a document."""


class UnsupportedFormatError(DocumentLoadError):
    """Document format is not supported."""


class EmbeddingError(NexusError):
    """Failed to generate embeddings."""


class VectorStoreError(NexusError):
    """Vector store operation failed."""


class RetrievalError(NexusError):
    """Retrieval operation failed."""


class GenerationError(NexusError):
    """LLM generation failed."""


class ConfigurationError(NexusError):
    """Invalid or missing configuration."""
