"""Set up a self-hosted Langfuse on this PC, to look at the app's traces (see src/ragapp/tracing.py).

    uv run python infra/langfuse/setup.py      # once: download Langfuse's compose file, write the secrets
    cd infra/langfuse && docker compose up -d  # start (about 7 GB RAM free is plenty); `docker compose stop` to stop
    then open http://localhost:3000 and log in with LANGFUSE_INIT_USER_EMAIL / _PASSWORD from infra/langfuse/.env

Writes, all git-ignored:
  docker-compose.yml  Langfuse's own file at a pinned release, refused if its checksum differs
  .env                random secrets, and the project with its API keys (created at Langfuse's first start)
  app.env             the lines that send the app's traces to it: add them to the project's .env
An existing .env is never overwritten: its passwords are already inside the database volumes.
Self-hosted on purpose: traces hold course page ids and, with TRACE_CONTENT=1, students' questions.
"""

import base64
import hashlib
import secrets
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = "v4.88.0"
COMPOSE_URL = f"https://raw.githubusercontent.com/langfuse/langfuse/{RELEASE}/docker-compose.yml"
COMPOSE_SHA256 = "a1abd7f9985171529fc61e473bbf447dfb49f3cc1c49a8573fca0427d86357ee"
WEB_URL = "http://localhost:3000"


def make_env(token=secrets.token_hex) -> dict[str, str]:
    """Langfuse's secrets (the same rules as its docs' script) and a project with API keys, for its first start."""
    postgres, minio = token(16), token(16)
    return {
        "LANGFUSE_IMAGE_TAG": RELEASE.lstrip("v"),  # the server version that matches the compose file
        "TELEMETRY_ENABLED": "false",  # no usage statistics sent to Langfuse
        "NEXTAUTH_SECRET": token(32),
        "SALT": token(16),
        "ENCRYPTION_KEY": token(32),  # 64 hex characters
        "POSTGRES_PASSWORD": postgres,
        "DATABASE_URL": f"postgresql://postgres:{postgres}@postgres:5432/postgres",  # must hold the same password
        "CLICKHOUSE_PASSWORD": token(16),
        "REDIS_AUTH": token(16),
        "MINIO_ROOT_PASSWORD": minio,
        # the three S3 keys must equal the MinIO root password
        "LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY": minio,
        "LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY": minio,
        "LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY": minio,
        # headless initialisation: organisation, project, its API keys and one login, created on first start
        "LANGFUSE_INIT_ORG_ID": "ragapp",
        "LANGFUSE_INIT_ORG_NAME": "RAG study assistant",
        "LANGFUSE_INIT_PROJECT_ID": "ragapp",
        "LANGFUSE_INIT_PROJECT_NAME": "ragapp",
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": f"pk-lf-{token(16)}",
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": f"sk-lf-{token(16)}",
        "LANGFUSE_INIT_USER_EMAIL": "admin@ragapp.localhost",
        "LANGFUSE_INIT_USER_NAME": "admin",
        "LANGFUSE_INIT_USER_PASSWORD": token(16),
    }


def app_env(env: dict[str, str]) -> dict[str, str]:
    """The app's side: OTLP over HTTP to Langfuse, with the project's keys as Basic auth."""
    keys = f"{env['LANGFUSE_INIT_PROJECT_PUBLIC_KEY']}:{env['LANGFUSE_INIT_PROJECT_SECRET_KEY']}"
    auth = base64.b64encode(keys.encode()).decode()
    return {
        "TRACING": "1",
        "OTEL_EXPORTER_OTLP_ENDPOINT": f"{WEB_URL}/api/public/otel",
        "OTEL_EXPORTER_OTLP_HEADERS": f"Authorization=Basic%20{auth},x-langfuse-ingestion-version=4",
    }


def env_text(values: dict[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in values.items())  # no quotes, no spaces around "="


def read_env(path: Path) -> dict[str, str]:
    pairs = (line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line)
    return {key.strip(): value.strip() for key, value in pairs}


def check_compose(data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != COMPOSE_SHA256:
        raise ValueError(f"docker-compose.yml checksum {digest} is not the pinned {COMPOSE_SHA256} ({RELEASE})")


def main(folder: Path = HERE) -> None:
    compose = folder / "docker-compose.yml"
    if not compose.exists():
        with urllib.request.urlopen(COMPOSE_URL, timeout=30) as response:
            data = response.read()
        check_compose(data)
        compose.write_bytes(data)
        print(f"downloaded Langfuse's docker-compose.yml ({RELEASE})")
    else:
        check_compose(compose.read_bytes())

    env_path = folder / ".env"
    if env_path.exists():
        print(f"kept the existing {env_path.name}: its passwords are already in the data volumes")
        env = read_env(env_path)
    else:
        env = make_env()
        env_path.write_text(env_text(env), encoding="utf-8")
        print(f"wrote {env_path} (random secrets; the login is LANGFUSE_INIT_USER_EMAIL / _PASSWORD)")
    (folder / "app.env").write_text(env_text(app_env(env)), encoding="utf-8")
    print(f"wrote {folder / 'app.env'}: add its lines to the project's .env to send traces")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
