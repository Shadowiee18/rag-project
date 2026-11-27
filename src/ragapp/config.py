"""All settings in one place, so experiments change one line."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "adsai_pages.jsonl"


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into the environment. A variable that is already set wins.
    (A few lines instead of the python-dotenv package.)"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Secrets (the BUas LLM server's address and key) live in the git-ignored .env file.
load_env_file(PROJECT_ROOT / ".env")

# Ollama (runs locally on http://localhost:11434): embeddings always, the answering LLM when LLM_BACKEND=ollama
EMBED_MODEL = "qwen3-embedding:0.6b"

# The answering LLM (answers and query rewriting): "ollama" = local GPU, "buas" = the BUas LLM server
# (an OpenAI-compatible API hosted for BUas students; needs BUAS_LLM_BASE_URL and BUAS_LLM_API_KEY in .env).
LLM_BACKEND = os.getenv("LLM_BACKEND", "buas")  # buas since Phase 3: as correct, far fewer made-up claims (README)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b" if LLM_BACKEND == "ollama" else "qwen3.8-27b")
BUAS_LLM_BASE_URL = os.getenv("BUAS_LLM_BASE_URL", "")
BUAS_LLM_API_KEY = os.getenv("BUAS_LLM_API_KEY", "")
# 1: the answer prompt treats the calendar-based Student line as a source (see generate.py). Off until measured.
STUDENT_LINE_AS_SOURCE = os.getenv("STUDENT_LINE_AS_SOURCE", "0") == "1"
# The eval's local judge stays the same model whatever answers, so correctness numbers stay comparable.
JUDGE_MODEL = "qwen3:8b"

# Tracing (see tracing.py): every step of every question as an OpenTelemetry span, e.g. into a local Langfuse.
TRACING = os.getenv("TRACING", "0") == "1"
# Record questions, queries and answers in the spans (user content; off by default, as in OpenTelemetry's GenAI rules).
TRACE_CONTENT = os.getenv("TRACE_CONTENT", "0") == "1"

# Qdrant (started with `docker compose up -d`)
QDRANT_URL = "http://localhost:6333"
# Apply the corpus-EDA cleaning rules at ingest (see cleaning.py). Needs a re-ingest when changed.
# On by default since the Phase 2 dev comparison (see README): same retrieval, tidier context.
CLEAN_CORPUS = os.getenv("CLEAN_CORPUS", "1") == "1"
# Separate collections let both indexes exist side by side; the default follows CLEAN_CORPUS.
COLLECTION = os.getenv("COLLECTION", "adsai_chunks_clean" if CLEAN_CORPUS else "adsai_chunks")
# Leave chunks of pages tagged page_type="template" out of search (only exists in a cleaned index).
EXCLUDE_TEMPLATES = os.getenv("EXCLUDE_TEMPLATES", "1") == "1"

# Chunking (characters). ~1500 chars is roughly 350-400 tokens.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Put "Year 2 · Block B · <page title>" at the top of every chunk before embedding,
# so a chunk still says where it comes from. Needs a re-ingest when changed.
# On by default since experiment 1 (see README results). Turn off for comparison:
#   CHUNK_HEADERS=0 uv run python -m ragapp.ingest
CHUNK_HEADERS = os.getenv("CHUNK_HEADERS", "1") == "1"

# Retrieval
TOP_K = 5

# Hybrid search (see bm25.py): vector search (meaning) + BM25 keyword search (exact rare words), merged with RRF.
# Needs `python -m ragapp.bm25` once after every ingest.
# On by default since the hybrid comparison (see README): page-style hit@1 64% -> 75% without a reranker, +0.3 s.
HYBRID = os.getenv("HYBRID", "1") == "1"
BM25_MODEL = "Qdrant/bm25"  # 0.01 GB: tokenizer, English stemmer and stopwords; Qdrant computes the IDF

# If the question names a year/block ("Year 2 Block B ..."), only search that year/block
# plus pages that belong to no year (all the shared Study Content).
# On by default since experiment 2 (see README results).
FILTER_BY_QUESTION_YEAR = os.getenv("FILTER_BY_QUESTION_YEAR", "1") == "1"
# How the question's scope (the asker's year/block, or the block the question names or "next block" points to;
# see retrieve.filter_for) is used, for every question alike:
#   "filter" search only the scope + shared pages (since experiment 2)
#   "boost"  that search AND one over the scope's own pages only, merge, rerank, then add a bonus to pages in the
#            scope (scores scaled 0-1 per question first). Shared pages get no bonus and no penalty.
SCOPE_MODE = os.getenv("SCOPE_MODE", "filter")
SCOPE_BOOST_BLOCK = float(os.getenv("SCOPE_BOOST_BLOCK", "0.2"))  # page in the scope's year and block
SCOPE_BOOST_YEAR = float(os.getenv("SCOPE_BOOST_YEAR", "0.0"))    # page in the scope's year

# Rewrite the question before searching (see rewrite.py):
#   "0" off | "1" v1, LLM writes one query (rejected by the eval) | "2" v2, LLM expansion + student week, multi-query + RRF
#   | "3" student week only: no LLM call, the original question plus a week-tagged copy for time questions
# "2" by default since the reranking comparison: alone it was noise, with a reranker student hit@5 went 47% -> 73%.
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "2")

# Rerank (see rerank.py): vector search finds RERANK_CANDIDATES chunks, a cross-encoder re-sorts them.
# "" = off. Models measured: "Xenova/ms-marco-MiniLM-L-6-v2" (0.08 GB), "BAAI/bge-reranker-base" (1.04 GB).
# bge-reranker-base: page hit@1 64% -> 96% on dev, but ~6.5 s per question on CPU (30 candidates); used when
# RERANK_KIND="cross" (below), which is no longer the default.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
# Words of each chunk the reranker reads (0 = the whole chunk, ~250-300 words). Fewer = faster reranking.
RERANK_MAX_WORDS = int(os.getenv("RERANK_MAX_WORDS", "0"))
# Use the int8 copy of the reranker made by `python -m ragapp.quantize` (faster on CPU).
RERANK_INT8 = os.getenv("RERANK_INT8", "0") == "1"
# Which reranker: "cross" (cross-encoder RERANK_MODEL) | "colbert" (late interaction scored inside Qdrant,
# needs `python -m ragapp.colbert` once after every ingest; see colbert.py).
# "colbert" by default since the hybrid comparison: with hybrid search ~3.7 s per answer, within 1-2 questions of
# the full cross-encoder's retrieval at 9.5 s. (RERANK_MODEL must stay non-empty: "" switches reranking off.)
RERANK_KIND = os.getenv("RERANK_KIND", "colbert")
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"  # 0.13 GB, 96 numbers per word piece
COLBERT_COLLECTION = f"{COLLECTION}_colbert"
BM25_COLLECTION = f"{COLLECTION}_bm25"
# Downloaded model files (git-ignored). fastembed's default is the temp folder, which Windows cleans up.
MODEL_CACHE = PROJECT_ROOT / "models"
