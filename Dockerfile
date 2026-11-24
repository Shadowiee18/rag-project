# 1. Start from an official, small Python image.
FROM python:3.13-slim

# 2. Get the uv binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

WORKDIR /app

# 3. Install dependencies first. Docker caches each step, so as long as these
#    two files don't change, rebuilding after a code change skips this slow step.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 4. Copy the code.
COPY phase0/exercises ./phase0/exercises

# Use the virtual environment uv created, so `uvicorn` is found.
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app/phase0/exercises
EXPOSE 8000

# 0.0.0.0 = accept connections from outside the container (127.0.0.1 would not).
CMD ["uvicorn", "ex06_api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
