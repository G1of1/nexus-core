"""In-memory vector store for testing and development."""

import math

from nexus.models.chunk import Chunk
from nexus.models.search import SearchFilter, SearchQuery, SearchResult
from nexus.providers.base import VectorStore


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _matches_filter(metadata: dict, filt: SearchFilter) -> bool:
    value = metadata.get(filt.field)
    if filt.operator == "eq":
        return value == filt.value
    if filt.operator == "ne":
        return value != filt.value
    if filt.operator == "in":
        return value in filt.value
    if filt.operator == "gt":
        return value is not None and value > filt.value
    if filt.operator == "lt":
        return value is not None and value < filt.value
    if filt.operator == "gte":
        return value is not None and value >= filt.value
    if filt.operator == "lte":
        return value is not None and value <= filt.value
    return False


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using cosine similarity."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    async def insert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    async def search(self, query: SearchQuery, query_vector: list[float]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for chunk in self._chunks.values():
            if chunk.embedding is None:
                continue
            if query.filters and not all(
                _matches_filter(chunk.metadata, f) for f in query.filters
            ):
                continue
            score = _cosine_similarity(query_vector, chunk.embedding)
            if query.score_threshold is not None and score < query.score_threshold:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    metadata=chunk.metadata,
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.top_k]

    async def delete(self, document_id: str) -> int:
        to_delete = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
        for cid in to_delete:
            del self._chunks[cid]
        return len(to_delete)

    async def delete_collection(self) -> None:
        self._chunks.clear()

    async def ensure_collection(self) -> None:
        pass
