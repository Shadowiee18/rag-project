import numpy as np

import ragapp.bm25 as bm25
import ragapp.retrieve as retrieve
from ragapp import config


def chunk(doc, i=0):
    return {"doc_id": doc, "chunk_index": i, "text": doc, "score": 0.5, "point_id": hash((doc, i)) % 1000}


def fake_search_setup(monkeypatch, dense_docs, keyword_docs, seen):
    monkeypatch.setattr(retrieve, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(retrieve, "get_client", lambda: None)
    monkeypatch.setattr(retrieve, "search", lambda client, v, k, year=None, blocks=None, shared=True: [chunk(d) for d in dense_docs][:k])

    def fake_bm25(question, k, year=None, blocks=None, shared=True):
        seen.append((question, year, blocks))
        return [chunk(d) for d in keyword_docs][:k]

    monkeypatch.setattr(retrieve, "bm25_search", fake_bm25)


def test_hybrid_adds_a_chunk_only_keyword_search_found(monkeypatch):
    seen = []
    fake_search_setup(monkeypatch, ["A", "B", "C"], ["K", "A"], seen)
    monkeypatch.setattr(config, "HYBRID", True)
    docs = [c["doc_id"] for c in retrieve.retrieve("keras backend", k=4, user_year="Year2", user_block="BlockB")]
    assert docs[0] == "A"  # found by both
    assert "K" in docs  # only BM25 found it
    assert seen == [("keras backend", "Year2", ["BlockB"])]  # same year/block filter as the vector search


def test_hybrid_off_is_vector_search_only(monkeypatch):
    seen = []
    fake_search_setup(monkeypatch, ["A", "B"], ["K"], seen)
    monkeypatch.setattr(config, "HYBRID", False)
    assert [c["doc_id"] for c in retrieve.retrieve("q", k=2)] == ["A", "B"]
    assert seen == []


def test_hybrid_runs_bm25_for_every_rewritten_query(monkeypatch):
    seen = []
    fake_search_setup(monkeypatch, ["A"], ["K"], seen)
    monkeypatch.setattr(config, "HYBRID", True)
    retrieve.retrieve_many(["is llm allowed", "is generative AI allowed"], 5, "is llm allowed")
    assert [q for q, _, _ in seen] == ["is llm allowed", "is generative AI allowed"]


def ranked_lists_sharing_one_deep_chunk(depth_rank):
    """Dense and BM25 lists with different chunks, except 'X' at the same deep rank in both.
    Fused, X (found twice) beats every chunk found once, but only if the lists are deep enough to contain it."""
    dense = [f"d{i}" for i in range(1, 30)]
    keyword = [f"k{i}" for i in range(1, 30)]
    dense[depth_rank - 1] = keyword[depth_rank - 1] = "X"
    return dense, keyword


def test_hybrid_top_chunks_do_not_depend_on_how_many_are_asked_for(monkeypatch):
    """The app asks for 5 chunks, the eval for 10: the eval's first 5 must be what the app gives the LLM."""
    fake_search_setup(monkeypatch, *ranked_lists_sharing_one_deep_chunk(7), [])
    monkeypatch.setattr(config, "HYBRID", True)
    app = [c["doc_id"] for c in retrieve.retrieve("q", k=5)]
    evaluation = [c["doc_id"] for c in retrieve.retrieve("q", k=10)]
    assert app == evaluation[:5]


def test_multi_query_top_chunks_do_not_depend_on_how_many_are_asked_for(monkeypatch):
    fake_search_setup(monkeypatch, *ranked_lists_sharing_one_deep_chunk(12), [])
    monkeypatch.setattr(config, "HYBRID", True)
    app = [c["doc_id"] for c in retrieve.retrieve_many(["q", "q2"], 5, "q")]
    evaluation = [c["doc_id"] for c in retrieve.retrieve_many(["q", "q2"], 10, "q")]
    assert app == evaluation[:5]


def test_bm25_search_with_only_stopwords_returns_nothing(monkeypatch):
    class EmptyQuery:
        indices = np.array([], dtype=int)
        values = np.array([])

    monkeypatch.setattr(bm25, "load_bm25", lambda: type("M", (), {"query_embed": lambda self, q: iter([EmptyQuery()])})())
    monkeypatch.setattr(bm25, "get_client", lambda: (_ for _ in ()).throw(AssertionError("must not query Qdrant")))
    assert bm25.bm25_search("is it", 5) == []
