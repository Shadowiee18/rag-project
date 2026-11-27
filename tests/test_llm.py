"""The answering LLM can run locally (Ollama) or on the BUas LLM server. No network or model is used here."""

import os

import pytest

import ragapp.generate as generate
import ragapp.llm as llm
from ragapp import config

MESSAGES = [{"role": "user", "content": "hi"}]


class _OllamaReply:
    def __init__(self, text):
        self.message = type("m", (), {"content": text})


class _HttpReply:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def test_ollama_backend(monkeypatch):
    seen = {}
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_MODEL", "qwen3:8b")
    monkeypatch.setattr(llm.ollama, "chat", lambda **kw: seen.update(kw) or _OllamaReply(" answer "))
    assert llm.chat(MESSAGES) == "answer"
    assert seen["model"] == "qwen3:8b" and seen["messages"] == MESSAGES
    assert seen["think"] is False and seen["options"]["temperature"] == 0


def _buas(monkeypatch, reply):
    seen = {}
    monkeypatch.setattr(config, "LLM_BACKEND", "buas")
    monkeypatch.setattr(config, "LLM_MODEL", "Some-Model")
    monkeypatch.setattr(config, "BUAS_LLM_BASE_URL", "https://llm.example/v1/")
    monkeypatch.setattr(config, "BUAS_LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm.httpx, "post", lambda url, **kw: seen.update(url=url, **kw) or reply)
    return seen


def test_buas_backend_sends_an_openai_style_request(monkeypatch):
    seen = _buas(monkeypatch, _HttpReply(200, {"choices": [{"message": {"content": " answer "}}]}))
    assert llm.chat(MESSAGES) == "answer"
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    body = seen["json"]
    assert body["model"] == "Some-Model" and body["messages"] == MESSAGES and body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}  # same choice as think=False locally


def test_buas_empty_content_is_an_empty_answer(monkeypatch):
    _buas(monkeypatch, _HttpReply(200, {"choices": [{"message": {"content": None}}]}))
    assert llm.chat(MESSAGES) == ""


def test_buas_error_is_raised_not_hidden(monkeypatch):
    """A model the server is not running must fail loudly, never fall back to another model quietly."""
    _buas(monkeypatch, _HttpReply(404, {"detail": "No active server provides model 'Some-Model'"}))
    with pytest.raises(RuntimeError, match="404.*No active server"):
        llm.chat(MESSAGES)


def test_buas_without_a_key_says_where_to_put_it(monkeypatch):
    _buas(monkeypatch, None)
    monkeypatch.setattr(config, "BUAS_LLM_API_KEY", "")
    with pytest.raises(RuntimeError, match=r"\.env"):
        llm.chat(MESSAGES)


def test_unknown_backend(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "cloud")
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        llm.chat(MESSAGES)


def test_generate_sends_system_and_user_messages(monkeypatch):
    seen = []
    monkeypatch.setattr(generate.llm, "chat", lambda messages: seen.append(messages) or "It is due Friday [1].")
    answer = generate.generate("when is it due", [{"title": "T", "text": "due Friday"}])
    assert answer == "It is due Friday [1]."
    assert [m["role"] for m in seen[0]] == ["system", "user"]
    assert seen[0][0]["content"] == generate.SYSTEM_PROMPT


def test_env_file_is_read_without_overriding_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# a comment\n\nRAG_TEST_A=one\nRAG_TEST_B = "two"\nRAG_TEST_C=three=3\nnot a setting\n')
    monkeypatch.setenv("RAG_TEST_A", "already set")
    monkeypatch.delenv("RAG_TEST_B", raising=False)
    monkeypatch.delenv("RAG_TEST_C", raising=False)
    config.load_env_file(env)
    assert os.environ["RAG_TEST_A"] == "already set"
    assert os.environ["RAG_TEST_B"] == "two"
    assert os.environ["RAG_TEST_C"] == "three=3"
    monkeypatch.delenv("RAG_TEST_B")
    monkeypatch.delenv("RAG_TEST_C")


def test_missing_env_file_is_fine(tmp_path):
    config.load_env_file(tmp_path / "nothing.env")
