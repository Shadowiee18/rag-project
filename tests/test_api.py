"""The HTTP API, with the RAG pipeline faked (no Ollama, no Qdrant)."""

from datetime import date

from fastapi.testclient import TestClient

import ragapp.api as api

client = TestClient(api.app)


def fake_pipeline(monkeypatch, seen: dict) -> None:
    def fake_answer(question, k, student):
        seen.update(question=question, k=k, student=student)
        return {"question": question, "answer": "Friday [1]", "chunks": [{"text": "full chunk text"}], "sources": []}

    monkeypatch.setattr(api, "answer", fake_answer)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_passes_the_student_on_and_returns_no_chunk_texts(monkeypatch):
    seen = {}
    fake_pipeline(monkeypatch, seen)
    response = client.get("/ask", params={"q": "whats due", "year": "Year2", "block": "BlockB", "today": "2026-11-24"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Friday [1]"
    assert "chunks" not in response.json()  # full chunk texts are for the eval, not for clients
    assert (seen["student"].year, seen["student"].block, seen["student"].today) == ("Year2", "BlockB", date(2026, 11, 24))


def test_ask_without_a_year_has_no_student_context(monkeypatch):
    seen = {}
    fake_pipeline(monkeypatch, seen)
    client.get("/ask", params={"q": "vpn"})
    assert seen["student"] is None


def test_ask_rejects_an_unknown_block():
    assert client.get("/ask", params={"q": "vpn", "year": "Year2", "block": "BlockZ"}).status_code == 422
