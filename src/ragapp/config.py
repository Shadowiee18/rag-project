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
#   "0" off | "1" v1, LLM writes one query (rejected by the eval) | "2" v2, multi-query + RRF
# Off by default: on dev, v2 moved every metric by about one question either way (noise level) and adds an LLM call.
QUERY_REWRITE = os.getenv("QUERY_REWRITE", "0")
