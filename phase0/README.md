# Phase 0: Refresher

Each exercise is a Python file in `exercises/` with function signatures and a docstring that
describes what the code must do. The bodies say `raise NotImplementedError`. Replace them with your code.
The tests in `tests/` tell you when you're done.

Every exercise here is something you'll reuse in the RAG system later.

```
uv run pytest -k ex01      # run one exercise
uv run pytest              # run everything
uv run pytest -x -q        # stop at first failure, short output
```

All tests fail at the start. That's expected.

## Step 0: git (do this first)

1. `git init`, then commit the starting files.
2. Create an empty GitHub repo and push to it.
3. After each exercise passes: `git add`, `git commit -m "ex01: word frequency"`, `git push`.

## Exercises

| # | File | You practise | Used later in |
|---|---|---|---|
| 01 | `ex01_word_freq.py` | strings, regex, dicts, sorting | tokenizing for BM25 |
| 02 | `ex02_chunking.py` | slicing, loops, input validation | splitting documents |
| 03 | `ex03_similarity.py` | math in plain Python, `zip`, sorting by key | vector search |
| 04 | `ex04_bm25.py` | classes, precomputing, `math.log` | keyword half of hybrid search |
| 05 | `ex05_documents.py` | pydantic models, file I/O, JSONL, `pathlib` | loading data and eval sets |
| 06 | `ex06_api.py` | FastAPI, request/response models, status codes | the app's API |
| 07 | Docker (below) | Dockerfile, ports, `uvicorn` | deployment in Phase 4 |

**Bonus after 03:** rewrite `top_k` with numpy (`uv add numpy`) and time both versions on 10,000 random vectors.

## 07: Docker (no tests; you check it by hand)

Write a `Dockerfile` at the project root that runs your ex06 app:

1. Start from an official slim Python image.
2. Install uv, copy `pyproject.toml` and `uv.lock`, run `uv sync --no-dev`.
3. Copy the `phase0/exercises` folder in.
4. From that folder, run `uv run --no-dev uvicorn ex06_api:create_app --factory --host 0.0.0.0 --port 8000`.
   (`0.0.0.0` matters: with the default `127.0.0.1`, the app can't be reached from outside the container.)

It works when this returns `{"status":"ok"}`:

```
docker build -t phase0-api .
docker run --rm -p 8000:8000 phase0-api
curl http://localhost:8000/health
```

Also add a `.dockerignore` so `.venv` doesn't get copied into the image.

## When you're stuck

Re-read the docstring and the failing test first. Then ask for a *hint*, not the answer.
After it passes, ask an AI to review your solution and explain one thing it would change and why.
