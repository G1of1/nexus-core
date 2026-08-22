"""Qdrant vector store provider."""

from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from nexus.config import NexusSettings
from nexus.exceptions import VectorStoreError
from nexus.models.chunk import Chunk
from nexus.models.search import SearchFilter, SearchQuery, SearchResult
from nexus.providers.base import VectorStore


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert chunk_id to a valid Qdrant point ID (UUID format)."""
    try:
        UUID(chunk_id)
        return chunk_id
    except ValueError:
        # Deterministic UUID from chunk_id string
        import hashlib

        hex_digest = hashlib.md5(chunk_id.encode()).hexdigest()
        return str(UUID(hex=hex_digest))


def _build_qdrant_filter(filters: list[SearchFilter]) -> models.Filter | None:
    if not filters:
        return None

    conditions: list[models.Condition] = []
    for f in filters:
        key = f.field
        if f.operator == "eq":
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=f.value))
            )
        elif f.operator == "ne":
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchExcept(**{"except": [f.value]}),
                )
            )
        elif f.operator == "in":
            conditions.append(
                models.FieldCondition(key=key, match=models.MatchAny(any=f.value))
            )
        elif f.operator in ("gt", "lt", "gte", "lte"):
            range_kwargs: dict[str, Any] = {}
            if f.operator in ("gt", "gte"):
                range_kwargs["gt" if f.operator == "gt" else "gte"] = f.value
            else:
                range_kwargs["lt" if f.operator == "lt" else "lte"] = f.value
            conditions.append(
                models.FieldCondition(key=key, range=models.Range(**range_kwargs))
            )

    return models.Filter(must=conditions) if conditions else None


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant."""

    def __init__(self, settings: NexusSettings) -> None:
        self._settings = settings
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._collection = settings.qdrant_collection

    async def ensure_collection(self) -> None:
        collections = await self._client.get_collections()
        exists = any(c.name == self._collection for c in collections.collections)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._settings.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    async def insert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        await self.ensure_collection()
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
            points.append(
                models.PointStruct(
                    id=_chunk_id_to_uuid(chunk.chunk_id),
                    vector=chunk.embedding,
                    payload=payload,
                )
            )
        try:
            await self._client.upsert(collection_name=self._collection, points=points)
        except Exception as e:
            raise VectorStoreError(f"Failed to insert chunks: {e}") from e

    async def search(self, query: SearchQuery, query_vector: list[float]) -> list[SearchResult]:
        qdrant_filter = _build_qdrant_filter(query.filters)
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=query.top_k,
                query_filter=qdrant_filter,
                score_threshold=query.score_threshold,
                with_payload=True,
            )
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}") from e

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    document_id=str(payload.get("document_id", "")),
                    content=str(payload.get("content", "")),
                    score=point.score,
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ("chunk_id", "document_id", "content", "chunk_index")
                    },
                )
            )
        return results

    async def delete(self, document_id: str) -> int:
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
            )
            return 0  # Qdrant delete doesn't return count
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document {document_id}: {e}") from e

    async def delete_collection(self) -> None:
        try:
            await self._client.delete_collection(collection_name=self._collection)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete collection: {e}") from e
