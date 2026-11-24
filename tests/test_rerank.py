import ragapp.rag as rag
import ragapp.rerank as rerank_module
from ragapp import config


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
