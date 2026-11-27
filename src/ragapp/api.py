"""HTTP API for the RAG system.

Run:  uv run uvicorn ragapp.api:app --reload    then open http://127.0.0.1:8000/docs
"""

from datetime import date
from typing import Annotated, Literal

from fastapi import FastAPI, Query

from ragapp.rag import answer
from ragapp.student import StudentContext
from ragapp.tracing import setup as setup_tracing

app = FastAPI(title="ADS & AI study assistant")
setup_tracing()  # TRACING=1: every /ask becomes a trace (see tracing.py)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask(
    q: Annotated[str, Query(min_length=1)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    year: Literal["Year1", "Year2", "Year3"] | None = None,
    block: Literal["BlockA", "BlockB", "BlockC", "BlockD", "Semester1", "Semester2"] | None = None,
    today: date | None = None,
):
    """year/block: the student's own. today: defaults to the real date. Used when the question doesn't say."""
    student = StudentContext(today=today or date.today(), year=year or "", block=block or "") if year else None
    result = answer(q, k, student)
    result.pop("chunks")  # the full chunk texts are for evaluation, not for API clients
    return result
