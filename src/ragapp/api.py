"""HTTP API for the RAG system (same ideas as phase0 ex06).

Run:  uv run uvicorn ragapp.api:app --reload    then open http://127.0.0.1:8000/docs
"""

from typing import Annotated, Literal

from fastapi import FastAPI, Query

from ragapp.rag import answer

app = FastAPI(title="ADS & AI study assistant")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask(
    q: Annotated[str, Query(min_length=1)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    year: Literal["Year1", "Year2", "Year3"] | None = None,
):
    """year: the student's own year. Used only when the question doesn't name a year itself."""
    return answer(q, k, user_year=year)
