"""infra/langfuse/setup.py: the secrets follow Langfuse's rules, the app's header is right, nothing is overwritten.
No network or Docker needed."""

import base64
import importlib.util
from itertools import count
from pathlib import Path
from urllib.parse import unquote

import pytest
from opentelemetry.util.re import parse_env_headers

SPEC = importlib.util.spec_from_file_location("langfuse_setup", Path(__file__).resolve().parents[1] / "infra" / "langfuse" / "setup.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def numbered_tokens():
    n = count()
    return lambda nbytes: f"{next(n):0{2 * nbytes}x}"  # distinct, with the right length


def test_secrets_follow_langfuses_rules():
    env = setup.make_env(numbered_tokens())
    assert env["POSTGRES_PASSWORD"] in env["DATABASE_URL"]
    for key in ("EVENT_UPLOAD", "MEDIA_UPLOAD", "BATCH_EXPORT"):
        assert env[f"LANGFUSE_S3_{key}_SECRET_ACCESS_KEY"] == env["MINIO_ROOT_PASSWORD"]
    assert len(env["ENCRYPTION_KEY"]) == 64
    assert env["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"].startswith("pk-lf-")
    assert env["LANGFUSE_INIT_PROJECT_SECRET_KEY"].startswith("sk-lf-")
    assert env["TELEMETRY_ENABLED"] == "false"
    assert env["LANGFUSE_IMAGE_TAG"] == "4.88.0"


def test_real_secrets_are_random_and_distinct():
    a, b = setup.make_env(), setup.make_env()
    assert a["NEXTAUTH_SECRET"] != b["NEXTAUTH_SECRET"]
    assert len({a["SALT"], a["REDIS_AUTH"], a["CLICKHOUSE_PASSWORD"], a["POSTGRES_PASSWORD"], a["MINIO_ROOT_PASSWORD"]}) == 5


def test_app_header_is_basic_auth_with_the_project_keys():
    env = setup.make_env(numbered_tokens())
    app = setup.app_env(env)
    assert app["TRACING"] == "1"
    assert app["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:3000/api/public/otel"
    headers = parse_env_headers(app["OTEL_EXPORTER_OTLP_HEADERS"], liberal=True)  # how the exporter reads it
    scheme, encoded = unquote(headers["authorization"]).split(" ")
    assert scheme == "Basic"
    assert base64.b64decode(encoded).decode() == f"{env['LANGFUSE_INIT_PROJECT_PUBLIC_KEY']}:{env['LANGFUSE_INIT_PROJECT_SECRET_KEY']}"
    assert headers["x-langfuse-ingestion-version"] == "4"
    assert (app["LANGFUSE_PUBLIC_KEY"], app["LANGFUSE_SECRET_KEY"]) == (
        env["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"], env["LANGFUSE_INIT_PROJECT_SECRET_KEY"])


def test_env_file_has_no_quotes_or_spaces_around_equals():
    env = setup.make_env()
    path_text = setup.env_text(env)
    assert '"' not in path_text and " = " not in path_text


def test_a_changed_compose_file_is_refused(tmp_path):
    (tmp_path / "docker-compose.yml").write_bytes(b"x")
    with pytest.raises(ValueError, match="checksum"):  # a changed compose file is refused
        setup.main(tmp_path)


def test_main_keeps_an_existing_env(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "check_compose", lambda data: None)
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    old = setup.make_env()
    (tmp_path / ".env").write_text(setup.env_text(old), encoding="utf-8")
    setup.main(tmp_path)
    assert setup.read_env(tmp_path / ".env") == old  # untouched: its passwords are in the volumes
    assert setup.read_env(tmp_path / "app.env") == setup.app_env(old)
