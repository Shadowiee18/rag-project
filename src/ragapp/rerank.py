"""Reranking: re-sort the retrieved chunks with a cross-encoder before the LLM sees them.

Vector search turns the question and each chunk into vectors separately and compares them: fast, but it can't
see which words of the chunk answer which words of the question. A cross-encoder reads question + chunk
together and scores how well the chunk answers it: more accurate, but far too slow for all ~3800 chunks.
So: vector search picks RERANK_CANDIDATES chunks, the cross-encoder re-sorts only those.
"""

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ragapp import config


@lru_cache(maxsize=1)  # load the model once per process, not once per question
def load_model(name: str) -> TextCrossEncoder:
    return TextCrossEncoder(name, cache_dir=str(config.MODEL_CACHE))


def rerank(question: str, chunks: list[dict]) -> list[dict]:
    """The same chunks, best first by cross-encoder score (which replaces the vector score)."""
    if not chunks:
        return []
    scores = list(load_model(config.RERANK_MODEL).rerank(question, [c["text"] for c in chunks]))
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [{**chunks[i], "score": float(scores[i])} for i in order]
