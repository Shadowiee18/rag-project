"""HTTP API for the RAG system.

Run:  uv run uvicorn ragapp.api:app --reload    then open http://127.0.0.1:8000 (chat) or /docs (the API)

Each browser gets a random id in a cookie the first time it saves a study year (PUT /profile).
The year is kept on the server (profile_store.py); the block and week come from today's date.
"""

import re
import secrets
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ragapp import config
from ragapp.profile import academic_year, status, student_context
from ragapp.profile_store import ProfileStore
from ragapp.rag import answer

app = FastAPI(title="ADS & AI study assistant")

COOKIE = "ragapp_user"
USER_ID = re.compile(r"[A-Za-z0-9_-]{22}")  # what secrets.token_urlsafe(16) makes: 128 random bits
COOKIE_DAYS = 400  # the longest browsers keep a cookie

Year = Literal["Year1", "Year2", "Year3"]
Block = Literal["BlockA", "BlockB", "BlockC", "BlockD", "Semester1", "Semester2"]


@lru_cache(maxsize=1)
def get_store() -> ProfileStore:
    return ProfileStore(config.PROFILE_DB)


Store = Annotated[ProfileStore, Depends(get_store)]


def _user_id(request: Request) -> str | None:
    """The browser's id, or None without a (well-formed) cookie."""
    value = request.cookies.get(COOKIE, "")
    return value if USER_ID.fullmatch(value) else None


def _saved(request: Request, store: ProfileStore) -> tuple[str, str]:
    """(year, academic year it was given in) from the saved profile; ("", "") without one."""
    user = _user_id(request)
    saved = store.get(user) if user else None
    return (saved.year, saved.given_in) if saved else ("", "")


@app.get("/", include_in_schema=False)
def chat_page():
    """The chat page: asks the study year once, then sends questions to /ask."""
    return FileResponse(Path(__file__).with_name("chat.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/profile")
def get_profile(request: Request, store: Store, today: date | None = None):
    """Whether to ask the study year (none saved, or saved in another academic year), and where the student is."""
    return status(*_saved(request, store), today or date.today())


class ProfileIn(BaseModel):
    year: Year


@app.put("/profile")
def put_profile(body: ProfileIn, request: Request, response: Response, store: Store, today: date | None = None):
    """Save the study year for the current academic year. today: defaults to the real date (for testing)."""
    today = today or date.today()
    user = _user_id(request) or secrets.token_urlsafe(16)
    store.save(user, body.year, academic_year(today))
    response.set_cookie(COOKIE, user, max_age=COOKIE_DAYS * 24 * 3600, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")
    return status(body.year, academic_year(today), today)


@app.get("/ask")
def ask(
    q: Annotated[str, Query(min_length=1)],
    request: Request,
    store: Store,
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    year: Year | None = None,
    block: Block | None = None,
    today: date | None = None,
):
    """The study year comes from the saved profile, the block from today's date.
    year/block override them (testing, evals). today: defaults to the real date.
    409 when the year must be asked first (none saved, or saved in an earlier academic year)."""
    today = today or date.today()
    year, given_in = (year, academic_year(today)) if year else _saved(request, store)
    where = status(year, given_in, today, block or "")
    if where["ask_year"]:
        raise HTTPException(409, "Study year unknown or from an earlier academic year: save it with PUT /profile.")
    result = answer(q, k, student_context(year, today, block or ""))
    result.pop("chunks")  # the full chunk texts are for evaluation, not for API clients
    return result | {"student": where}
