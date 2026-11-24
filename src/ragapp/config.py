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
COLLECTION = "adsai_chunks"

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
