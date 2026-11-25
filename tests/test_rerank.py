import pytest

import ragapp.rag as rag
import ragapp.rerank as rerank_module
from ragapp import config


@pytest.fixture(autouse=True)
def cross_encoder_kind(monkeypatch):
    """These tests are about the cross-encoder path; the ColBERT test switches the kind itself."""
    monkeypatch.setattr(config, "RERANK_KIND", "cross")


class FakeCrossEncoder:
    """Scores a chunk by how many question words it contains."""

    def rerank(self, query, documents):
        words = set(query.lower().split())
        return [float(len(words & set(d.lower().split()))) for d in documents]


def chunk(i, text):
    return {"doc_id": f"d{i}", "chunk_index": 0, "title": "T", "source": "s", "text": text, "score": 0.5}


def test_rerank_sorts_by_cross_encoder_score(monkeypatch):
    monkeypatch.setattr(rerank_module, "load_model", lambda name: FakeCrossEncoder())
    chunks = [chunk(0, "nothing here"), chunk(1, "vpn setup from home"), chunk(2, "vpn")]
    ranked = rerank_module.rerank("vpn setup", chunks)
    assert [c["doc_id"] for c in ranked] == ["d1", "d2", "d0"]
    assert ranked[0]["score"] == 2.0


def test_answer_reranks_a_bigger_pool_then_keeps_the_best(monkeypatch):
    seen = {}

    def fake_retrieve(question, k, user_year=None, user_block=None):
        seen["k"] = k
        return [chunk(i, "vpn" if i == 29 else "other") for i in range(k)]

    monkeypatch.setattr(config, "QUERY_REWRITE", "0")
    monkeypatch.setattr(config, "RERANK_MODEL", "fake")
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 30)
    monkeypatch.setattr(rerank_module, "load_model", lambda name: FakeCrossEncoder())
    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "generate", lambda question, chunks, student: "answer")

    out = rag.answer("vpn", k=5, retrieve_k=10)
    assert seen["k"] == 30  # searched 30, not 10
    assert len(out["chunks"]) == 10
    assert out["chunks"][0]["doc_id"] == "d29"  # the last vector hit became first


def test_reranking_off_searches_only_what_is_needed_and_loads_no_model(monkeypatch):
    seen = {}

    def fake_retrieve(question, k, user_year=None, user_block=None):
        seen["k"] = k
        return [chunk(i, "other") for i in range(k)]

    def no_model(name):
        raise AssertionError("the reranker must not load when RERANK_MODEL is empty")

    monkeypatch.setattr(config, "QUERY_REWRITE", "0")
    monkeypatch.setattr(config, "RERANK_MODEL", "")
    monkeypatch.setattr(rerank_module, "load_model", no_model)
    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "generate", lambda question, chunks, student: "answer")

    out = rag.answer("vpn", k=5, retrieve_k=10)
    assert seen["k"] == 10
    assert [c["doc_id"] for c in out["chunks"]] == [f"d{i}" for i in range(10)]  # vector order kept


def test_reranker_reads_only_the_first_words_but_returns_the_full_chunk(monkeypatch):
    read = []

    class Recorder:
        def rerank(self, query, documents):
            read.extend(documents)
            return [1.0 for _ in documents]

    monkeypatch.setattr(config, "RERANK_MAX_WORDS", 3)
    monkeypatch.setattr(rerank_module, "load_model", lambda name: Recorder())
    out = rerank_module.rerank("q", [chunk(0, "Year 2 · Block B one two three")])
    assert read == ["Year 2 ·"]
    assert out[0]["text"] == "Year 2 · Block B one two three"  # the LLM still gets everything


def test_max_words_zero_reads_the_whole_chunk(monkeypatch):
    monkeypatch.setattr(config, "RERANK_MAX_WORDS", 0)
    assert rerank_module.shorten("a b c d") == "a b c d"


def test_colbert_kind_reorders_by_the_scores_qdrant_returns(monkeypatch):
    import numpy as np

    import ragapp.colbert as colbert

    class FakeColbert:
        def query_embed(self, question):
            yield np.zeros((3, 4))  # 3 question word pieces, 4 numbers each

    class FakeClient:
        def query_points(self, collection_name, query, query_filter, limit):
            assert collection_name == config.COLBERT_COLLECTION
            assert sorted(query_filter.must[0].has_id) == [10, 11, 12]  # only the candidates are scored
            point = lambda i, s: type("P", (), {"id": i, "score": s})()  # noqa: E731
            return type("R", (), {"points": [point(12, 9.0), point(10, 5.0), point(11, 1.0)]})()

    monkeypatch.setattr(config, "RERANK_KIND", "colbert")
    monkeypatch.setattr(colbert, "load_colbert", lambda: FakeColbert())
    monkeypatch.setattr(colbert, "get_client", lambda: FakeClient())
    chunks = [{**chunk(i, "t"), "point_id": 10 + i} for i in range(3)]
    ranked = rerank_module.rerank("q", chunks)
    assert [c["point_id"] for c in ranked] == [12, 10, 11]
    assert ranked[0]["score"] == 9.0


def test_int8_loads_the_quantized_copy(monkeypatch):
    seen = {}
    monkeypatch.setattr(config, "RERANK_INT8", True)
    monkeypatch.setattr(rerank_module, "TextCrossEncoder", lambda name, **kw: seen.update(kw) or "model")
    rerank_module.load_model.cache_clear()
    try:
        rerank_module.load_model("org/name")
    finally:
        rerank_module.load_model.cache_clear()
    assert seen["specific_model_path"].endswith("org--name-int8")
