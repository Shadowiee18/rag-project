"""Exercise 06: a small search API with FastAPI.

This reuses your code from ex01 (tokenize), ex04 (BM25) and ex05 (Document).
Try it in a browser after the tests pass:

    cd phase0/exercises
    uv run uvicorn ex06_api:create_app --factory --reload

then open http://127.0.0.1:8000/docs

Run the tests:  uv run pytest -k ex06
"""

from fastapi import FastAPI

# from ex01_word_freq import tokenize
# from ex04_bm25 import BM25
# from ex05_documents import Document


def create_app() -> FastAPI:
    """Build and return the app.

    Keep the stored documents in a dict created inside this function, so each call to
    create_app() starts empty. (The tests rely on this.)

    Endpoints:

    GET  /health
        200 {"status": "ok"}

    POST /documents
        body: a Document (id, text, source, optional metadata)
        201 {"id": "<the id>"}
        409 if a document with that id is already stored
        422 if a field is missing or empty (pydantic does this for you)

    GET  /documents/{doc_id}
        200 the stored document as JSON
        404 if there is no such document

    GET  /search?q=<query>&k=<number>
        200 {"results": [{"id": "<doc id>", "score": <float>}, ...]}
        Rank stored documents with BM25 over tokenize(document.text), best first, at most k results.
        k defaults to 3 and must be between 1 and 20, otherwise 422.
        q must not be empty, otherwise 422.
        No stored documents, or no matches: {"results": []}
        Building a new BM25 index on every search is fine here.

    Hints: HTTPException, status_code=201 on the decorator, fastapi.Query(ge=..., le=..., min_length=...).
    """
    raise NotImplementedError
