"""BM25 keyword search, the second half of hybrid search.

Vector search finds chunks with similar MEANING, but blurs exact rare words ("keras", "task 8", "pyspark").
BM25 finds chunks that literally contain the question's words, and the rarer a word is in the corpus, the more
it counts (IDF). Hybrid search runs both and merges the lists with reciprocal rank fusion (see retrieve.py).

Each chunk is stored as a sparse vector: word ids with BM25 term-frequency weights (fastembed "Qdrant/bm25",
lower-cased, English stemming and stopwords). Qdrant adds the IDF part itself (Modifier.IDF), so the corpus
statistics stay correct. It lives in its own collection with the same point ids and payload (text, year, block,
page type) as the main one, so the year/block filter and template exclusion work exactly the same.

Build once, after every ingest:
    uv run python -m ragapp.bm25
"""

import time
from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client.models import Modifier, PointStruct, SparseVector, SparseVectorParams

from ragapp import config
from ragapp.store import build_filter, get_client

VECTOR_NAME = "bm25"
BATCH_SIZE = 256


@lru_cache(maxsize=1)  # load once per process
def load_bm25() -> SparseTextEmbedding:
    return SparseTextEmbedding(config.BM25_MODEL, cache_dir=str(config.MODEL_CACHE))


def sparse(embedding) -> SparseVector:
    return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


def bm25_search(question: str, k: int, year: str | None = None, blocks: list[str] | None = None) -> list[dict]:
    """The k chunks with the best BM25 score, as dicts like store.search returns (payload + score + point_id)."""
    query = next(iter(load_bm25().query_embed(question)))
    if len(query.indices) == 0:  # only stopwords ("is it?"): BM25 has nothing to match on
        return []
    result = get_client().query_points(
        collection_name=config.BM25_COLLECTION,
        query=sparse(query),
        using=VECTOR_NAME,
        query_filter=build_filter(year, blocks),
        limit=k,
        with_payload=True,
    )
    return [{**point.payload, "score": point.score, "point_id": point.id} for point in result.points]


def main() -> None:
    """Indexing time: a BM25 sparse vector for every chunk of config.COLLECTION, under the same point ids."""
    started = time.perf_counter()
    client = get_client()
    points, offset = [], None
    while True:  # every chunk with its full payload, from the main collection
        batch, offset = client.scroll(config.COLLECTION, limit=1000, offset=offset, with_payload=True)
        points.extend(batch)
        if offset is None:
            break
    print(f"{len(points)} chunks from '{config.COLLECTION}'")

    if client.collection_exists(config.BM25_COLLECTION):
        client.delete_collection(config.BM25_COLLECTION)
    client.create_collection(
        collection_name=config.BM25_COLLECTION,
        vectors_config={},  # no dense vectors here, only the sparse one
        sparse_vectors_config={VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )
    model = load_bm25()
    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start : start + BATCH_SIZE]
        vectors = model.embed([p.payload["text"] for p in batch])
        client.upsert(
            collection_name=config.BM25_COLLECTION,
            points=[PointStruct(id=p.id, vector={VECTOR_NAME: sparse(v)}, payload=p.payload) for p, v in zip(batch, vectors)],
        )
    print(f"done in {time.perf_counter() - started:.0f}s -> collection '{config.BM25_COLLECTION}'")


if __name__ == "__main__":
    main()
