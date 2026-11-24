"""Qdrant vector database: create the collection, store chunks, search."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams

from ragapp import config


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


def recreate_collection(client: QdrantClient, vector_size: int) -> None:
    """Start from an empty collection, so re-running ingest never leaves old chunks behind."""
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def add_chunks(client: QdrantClient, start_id: int, vectors: list[list[float]], payloads: list[dict]) -> None:
    points = [
        PointStruct(id=start_id + i, vector=vector, payload=payload)
        for i, (vector, payload) in enumerate(zip(vectors, payloads))
    ]
    client.upsert(collection_name=config.COLLECTION, points=points)


def year_block_filter(year: str, blocks: list[str] | None) -> Filter:
    """Keep chunks from this year (and these blocks), plus chunks that belong to no year/block.

    The "" option matters: shared Study Content pages have no year, and must stay searchable.
    """
    conditions = [FieldCondition(key="year", match=MatchAny(any=[year, ""]))]
    if blocks:
        conditions.append(FieldCondition(key="block", match=MatchAny(any=[*blocks, ""])))
    return Filter(must=conditions)


def search(
    client: QdrantClient, query_vector: list[float], k: int, year: str | None = None, blocks: list[str] | None = None
) -> list[dict]:
    """Return the k closest chunks as dicts: the stored payload plus a 'score'."""
    result = client.query_points(
        collection_name=config.COLLECTION,
        query=query_vector,
        query_filter=year_block_filter(year, blocks) if year else None,
        limit=k,
        with_payload=True,
    )
    return [{**point.payload, "score": point.score} for point in result.points]
