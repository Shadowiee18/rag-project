"""All settings in one place, so experiments change one line."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "adsai_pages.jsonl"

# Ollama (runs locally on http://localhost:11434)
EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3:8b"

# Qdrant (started with `docker compose up -d`)
QDRANT_URL = "http://localhost:6333"
# Apply the corpus-EDA cleaning rules at ingest (see cleaning.py). Needs a re-ingest when changed.
# On by default since the Phase 2 dev comparison (see README): same retrieval, tidier context.
CLEAN_CORPUS = os.getenv("CLEAN_CORPUS", "1") == "1"
# Separate collections let both indexes exist side by side; the default follows CLEAN_CORPUS.
COLLECTION = os.getenv("COLLECTION", "adsai_chunks_clean" if CLEAN_CORPUS else "adsai_chunks")
# Leave chunks of pages tagged page_type="template" out of search (only exists in a cleaned index).
EXCLUDE_TEMPLATES = os.getenv("EXCLUDE_TEMPLATES", "1") == "1"

# Chunking (characters, like phase0 ex02). ~1500 chars is roughly 350-400 tokens.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Put "Year 2 · Block B · <page title>" at the top of every chunk before embedding,
# so a chunk still says where it comes from. Needs a re-ingest when changed.
# On by default since experiment 1 (see README results). Turn off for comparison:
#   CHUNK_HEADERS=0 uv run python -m ragapp.ingest
CHUNK_HEADERS = os.getenv("CHUNK_HEADERS", "1") == "1"

# Retrieval
TOP_K = 5

# If the question names a year/block ("Year 2 Block B ..."), only search that year/block
# plus pages that belong to no year (all the shared Study Content).
# On by default since experiment 2 (see README results).
FILTER_BY_QUESTION_YEAR = os.getenv("FILTER_BY_QUESTION_YEAR", "1") == "1"

# Rewrite the question before searching (see rewrite.py):
#   "0" off | "1" v1, LLM writes one query (rejected by the eval) | "2" v2, LLM expansion + student week, multi-query + RRF
#   | "3" student week only: no LLM call, the original question plus a week-tagged copy for time questions
# "2" by default since the reranking comparison: alone it was noise, with a reranker student hit@5 went 47% -> 73%.
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "2")

# Rerank (see rerank.py): vector search finds RERANK_CANDIDATES chunks, a cross-encoder re-sorts them.
# "" = off. Models measured: "Xenova/ms-marco-MiniLM-L-6-v2" (0.08 GB), "BAAI/bge-reranker-base" (1.04 GB).
# bge-reranker-base by default: page hit@1 64% -> 96% on dev. Costs ~6.5 s per question on CPU (30 candidates).
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "30"))
# Words of each chunk the reranker reads (0 = the whole chunk, ~250-300 words). Fewer = faster reranking.
RERANK_MAX_WORDS = int(os.getenv("RERANK_MAX_WORDS", "0"))
# Use the int8 copy of the reranker made by `python -m ragapp.quantize` (faster on CPU).
RERANK_INT8 = os.getenv("RERANK_INT8", "0") == "1"
# Which reranker: "cross" (cross-encoder RERANK_MODEL) | "colbert" (late interaction scored inside Qdrant,
# needs `python -m ragapp.colbert` once after every ingest; see colbert.py).
RERANK_KIND = os.getenv("RERANK_KIND", "cross")
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"  # 0.13 GB, 96 numbers per word piece
COLBERT_COLLECTION = f"{COLLECTION}_colbert"
# Downloaded model files (git-ignored). fastembed's default is the temp folder, which Windows cleans up.
MODEL_CACHE = PROJECT_ROOT / "models"
