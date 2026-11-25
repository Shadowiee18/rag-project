"""Late-interaction (ColBERT) reranking: the heavy model work on chunks moves from query time to indexing time.

A cross-encoder reads question + chunk together, so every question needs a model run per candidate.
ColBERT encodes them separately, keeping one small vector per word piece (not one per chunk):
- indexing time (once): every chunk -> its word-piece vectors, stored in their own Qdrant collection
- query time: only the question goes through the model; Qdrant scores each candidate with MaxSim
  (for every question word, the best-matching chunk word; summed), so the rerank is a fast vector lookup.

Build once, after every ingest (it copies the point ids of the main collection):
    uv run python -m ragapp.colbert
"""

import time
from functools import lru_cache

from fastembed import LateInteractionTextEmbedding
from qdrant_client.models import (
    Distance,
    Filter,
    HasIdCondition,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

from ragapp import config
from ragapp.store import get_client

BATCH_SIZE = 32


@lru_cache(maxsize=1)  # load once per process
def load_colbert() -> LateInteractionTextEmbedding:
    return LateInteractionTextEmbedding(config.COLBERT_MODEL, cache_dir=str(config.MODEL_CACHE))


def colbert_rerank(question: str, chunks: list[dict]) -> list[dict]:
    """The same chunks, best first by MaxSim score (which replaces the vector score)."""
    if not chunks:
        return []
    query = next(iter(load_colbert().query_embed(question))).tolist()  # one vector per question word piece
    ids = [c["point_id"] for c in chunks]
    result = get_client().query_points(
        collection_name=config.COLBERT_COLLECTION,
        query=query,
        query_filter=Filter(must=[HasIdCondition(has_id=ids)]),  # score only the candidates
        limit=len(ids),
    )
    scores = {point.id: point.score for point in result.points}
    ranked = sorted(chunks, key=lambda c: -scores.get(c["point_id"], float("-inf")))
    return [{**c, "score": scores.get(c["point_id"], 0.0)} for c in ranked]


def main() -> None:
    """Indexing time: word-piece vectors for every chunk of config.COLLECTION, under the same point ids."""
    started = time.perf_counter()
    client = get_client()
    points, offset = [], None
    while True:  # read every chunk's text from the main collection
        batch, offset = client.scroll(config.COLLECTION, limit=1000, offset=offset, with_payload=["text"])
        points.extend(batch)
        if offset is None:
            break
    print(f"{len(points)} chunks from '{config.COLLECTION}'")

    model = load_colbert()
    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start : start + BATCH_SIZE]
        vectors = list(model.embed([p.payload["text"] for p in batch]))
        if start == 0:
            if client.collection_exists(config.COLBERT_COLLECTION):
                client.delete_collection(config.COLBERT_COLLECTION)
            client.create_collection(
                collection_name=config.COLBERT_COLLECTION,
                vectors_config=VectorParams(
                    size=vectors[0].shape[1],
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                    hnsw_config=HnswConfigDiff(m=0),  # no search index: only used to rescore known candidates
                ),
            )
            print(f"{vectors[0].shape[1]} numbers per word piece")
        client.upsert(
            collection_name=config.COLBERT_COLLECTION,
            points=[PointStruct(id=p.id, vector=v.tolist()) for p, v in zip(batch, vectors)],
        )
        print(f"  encoded {min(start + BATCH_SIZE, len(points))}/{len(points)}", end="\r")
    print(f"\ndone in {time.perf_counter() - started:.0f}s -> collection '{config.COLBERT_COLLECTION}'")


if __name__ == "__main__":
    main()
