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


def build_filter(year: str | None, blocks: list[str] | None, shared: bool = True) -> Filter | None:
    """Keep chunks from this year (and these blocks), plus chunks that belong to no year/block.
    Leave out template pages (if the index tags them and EXCLUDE_TEMPLATES is on).

    The "" option matters: shared Study Content pages have no year, and must stay searchable.
    shared=False: only the year's (and blocks') own pages, for the scoped search of SCOPE_MODE=boost.
    """
    must, must_not = [], []
    none = [""] if shared else []
    if year:
        must.append(FieldCondition(key="year", match=MatchAny(any=[year, *none])))
        if blocks:
            must.append(FieldCondition(key="block", match=MatchAny(any=[*blocks, *none])))
    if config.EXCLUDE_TEMPLATES:
        must_not.append(FieldCondition(key="page_type", match=MatchAny(any=["template"])))
    return Filter(must=must, must_not=must_not) if (must or must_not) else None


def search(
    client: QdrantClient,
    query_vector: list[float],
    k: int,
    year: str | None = None,
    blocks: list[str] | None = None,
    shared: bool = True,
) -> list[dict]:
    """Return the k closest chunks as dicts: the stored payload plus a 'score' and the Qdrant 'point_id'."""
    result = client.query_points(
        collection_name=config.COLLECTION,
        query=query_vector,
        query_filter=build_filter(year, blocks, shared),
        limit=k,
        with_payload=True,
    )
    return [{**point.payload, "score": point.score, "point_id": point.id} for point in result.points]
