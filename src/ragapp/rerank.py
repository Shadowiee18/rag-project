"""Reranking: re-sort the retrieved chunks with a cross-encoder before the LLM sees them.

Vector search turns the question and each chunk into vectors separately and compares them: fast, but it can't
see which words of the chunk answer which words of the question. A cross-encoder reads question + chunk
together and scores how well the chunk answers it: more accurate, but far too slow for all ~3800 chunks.
So: vector search picks RERANK_CANDIDATES chunks, the cross-encoder re-sorts only those.
"""

from functools import lru_cache
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ragapp import config
from ragapp.colbert import colbert_rerank


def int8_dir(name: str) -> Path:
    """Where `python -m ragapp.quantize` writes the int8 copy of a reranker."""
    return config.MODEL_CACHE / f"{name.replace('/', '--')}-int8"


@lru_cache(maxsize=1)  # load the model once per process, not once per question
def load_model(name: str) -> TextCrossEncoder:
    if config.RERANK_INT8:
        return TextCrossEncoder(name, cache_dir=str(config.MODEL_CACHE), specific_model_path=str(int8_dir(name)))
    return TextCrossEncoder(name, cache_dir=str(config.MODEL_CACHE))


def shorten(text: str) -> str:
    """What the cross-encoder reads: the first RERANK_MAX_WORDS words (0 = all). Its cost grows with length,
    so fewer words = faster. The chunk header ("Year 2 · Block B · title") is at the start, so it always stays."""
    if not config.RERANK_MAX_WORDS:
        return text
    return " ".join(text.split()[: config.RERANK_MAX_WORDS])


def rerank(question: str, chunks: list[dict]) -> list[dict]:
    """The same chunks, best first by cross-encoder score (which replaces the vector score).
    The LLM still gets the full chunk text; only the scoring reads the shortened version."""
    if config.RERANK_KIND == "colbert":
        return colbert_rerank(question, chunks)
    if not chunks:
        return []
    scores = list(load_model(config.RERANK_MODEL).rerank(question, [shorten(c["text"]) for c in chunks]))
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [{**chunks[i], "score": float(scores[i])} for i in order]
