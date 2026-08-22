# Nexus Core

Reusable RAG engine library for the Nexus platform. Independent of FastAPI and designed for dependency injection across APIs, workers, and CLI tools.

## Architecture

```
nexus/
├── models/          # Domain types (Document, Chunk, SearchResult)
├── providers/       # Abstract interfaces + concrete implementations
├── processing/      # Document loaders and chunking
├── retrieval/       # Vector search, filtering, reranking
├── generation/      # Prompt templates and LLM orchestration
├── pipeline/        # Ingestion and end-to-end RAG orchestration
├── evaluation/      # Retrieval and answer quality metrics
└── config.py        # Environment-based configuration
```

## Quick Start

```python
from nexus.config import NexusSettings
from nexus.pipeline.rag import RAGEngine
from nexus.providers.embeddings.gemini import GeminiEmbeddingProvider
from nexus.providers.llm.gemini import GeminiLLMProvider
from nexus.providers.vectorstore.qdrant import QdrantVectorStore

settings = NexusSettings()
engine = RAGEngine(
    vector_store=QdrantVectorStore(settings),
    embedding_provider=GeminiEmbeddingProvider(settings),
    llm_provider=GeminiLLMProvider(settings),
)

# Ingest
await engine.ingest_file("report.pdf", metadata={"source": "upload"})

# Query
response = await engine.query("What are the key findings?")
print(response.answer)
for citation in response.citations:
    print(f"  - {citation.source}: {citation.excerpt}")
```

## Configuration

All settings are loaded from environment variables with the `NEXUS_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXUS_QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `NEXUS_QDRANT_COLLECTION` | Vector collection name | `nexus_documents` |
| `NEXUS_GEMINI_API_KEY` | Google Gemini API key | — |
| `NEXUS_EMBEDDING_MODEL` | Embedding model | `models/text-embedding-004` |
| `NEXUS_LLM_MODEL` | Chat model | `gemini-2.0-flash` |
| `NEXUS_CHUNK_SIZE` | Chunk size in tokens | `512` |
| `NEXUS_CHUNK_OVERLAP` | Chunk overlap in tokens | `64` |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Design Principles

- **Framework-agnostic**: No FastAPI, Celery, or web dependencies in core.
- **Dependency injection**: All infrastructure accessed through provider interfaces.
- **Vendor abstraction**: Swap Qdrant, Gemini, or other backends without changing business logic.
- **Async-first**: Native async/await for I/O-bound operations.
