"""SCOPE_MODE=boost: the question's scope (the asker's year/block, or the block the question names or points to
with "next block") is a bonus in the ranking, never a filter.

One mechanism for every question: search everything AND the scope's own pages, merge, rerank, then add
w_block / w_year to pages in the scope. Shared pages are never removed, so they still win when they are the better
match. No Ollama, Qdrant or course data needed.
"""

from datetime import date

import pytest

import ragapp.rag as rag
import ragapp.retrieve as retrieve
from ragapp import config
from ragapp.retrieve import scope_boost
from ragapp.store import build_filter
from ragapp.student import StudentContext

TODAY = date(2026, 11, 26)  # Year 2 Block B week 3 on the calendar


def matches(f) -> dict[str, list[str]]:
    return {c.key: list(c.match.any) for c in f.must}


def page(doc_id, score, year="", block=""):
    return {"doc_id": doc_id, "chunk_index": 0, "score": score, "year": year, "block": block, "title": doc_id,
            "source": "", "text": doc_id, "point_id": doc_id}


# ---------------------------------------------------------------- the filter for the scoped search


def test_default_filter_keeps_the_shared_pages():
    assert matches(build_filter("Year2", ["BlockC"])) == {"year": ["Year2", ""], "block": ["BlockC", ""]}


def test_scope_only_filter_leaves_the_shared_pages_out():
    assert matches(build_filter("Year2", ["BlockC"], shared=False)) == {"year": ["Year2"], "block": ["BlockC"]}


def test_scope_only_filter_for_a_whole_year():
    assert matches(build_filter("Year1", None, shared=False)) == {"year": ["Year1"]}


# ---------------------------------------------------------------- which searches run


@pytest.fixture
def recorded(monkeypatch):
    """Replace the embedder and both searches; record (kind, year, blocks, shared) of every search."""
    calls = []
    monkeypatch.setattr(retrieve, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(retrieve, "get_client", lambda: None)
    monkeypatch.setattr(retrieve, "search", lambda client, v, k, year=None, blocks=None, shared=True:
                        calls.append(("dense", year, blocks, shared)) or [])
    monkeypatch.setattr(retrieve, "bm25_search", lambda q, k, year=None, blocks=None, shared=True:
                        calls.append(("bm25", year, blocks, shared)) or [])
    monkeypatch.setattr(config, "HYBRID", True)
    return calls


def same(calls: list, expected: list) -> bool:
    """The same searches, in any order."""
    return sorted(map(repr, calls)) == sorted(map(repr, expected))


def test_filter_mode_searches_as_before(recorded):
    retrieve.retrieve_many(["q1", "q2"], 5, "what do we have next block?", "Year2", "BlockB", TODAY)
    assert recorded == [("dense", "Year2", ["BlockC"], True)] * 2 + [("bm25", "Year2", ["BlockC"], True)] * 2


def test_boost_mode_searches_everything_and_the_scope(recorded, monkeypatch):
    monkeypatch.setattr(config, "SCOPE_MODE", "boost")
    retrieve.retrieve_many(["q1", "q2"], 5, "what do we have next block?", "Year2", "BlockB", TODAY)
    scoped = ("Year2", ["BlockC"], False)
    assert same(recorded, [("dense", None, None, True)] * 2 + [("bm25", None, None, True)] * 2
                + [("dense", *scoped)] * 2 + [("bm25", *scoped)] * 2)


def test_boost_mode_scope_is_the_askers_own_block_without_a_block_in_the_question(recorded, monkeypatch):
    monkeypatch.setattr(config, "SCOPE_MODE", "boost")
    retrieve.retrieve_many(["whats due this week"], 5, "whats due this week", "Year2", "BlockB", TODAY)
    assert ("dense", "Year2", ["BlockB"], False) in recorded


def test_boost_mode_without_any_scope_is_one_open_search(recorded, monkeypatch):
    monkeypatch.setattr(config, "SCOPE_MODE", "boost")
    retrieve.retrieve_many(["what is overfitting"], 5, "what is overfitting", None, None, TODAY)
    assert recorded == [("dense", None, None, True), ("bm25", None, None, True)]


def test_single_query_retrieve_uses_the_same_searches(recorded, monkeypatch):
    monkeypatch.setattr(config, "SCOPE_MODE", "boost")
    retrieve.retrieve("Year 1 Block C overview", 5, "Year2", "BlockB", TODAY)
    assert same(recorded, [("dense", None, None, True), ("bm25", None, None, True),
                           ("dense", "Year1", ["BlockC"], False), ("bm25", "Year1", ["BlockC"], False)])


# ---------------------------------------------------------------- the bonus


def ids(chunks):
    return [c["doc_id"] for c in chunks]


def test_bonus_lifts_a_scope_page_over_a_slightly_better_shared_page():
    chunks = [page("shared", 20.0), page("y2c", 19.0, "Year2", "BlockC"), page("other", 10.0)]
    assert ids(scope_boost(chunks, "Year2", ["BlockC"], w_block=0.2, w_year=0.0)) == ["y2c", "shared", "other"]


def test_a_much_better_shared_page_stays_first():
    """A bonus, not a filter: shared pages can still win."""
    chunks = [page("shared", 20.0), page("other", 15.0), page("y2c", 10.0, "Year2", "BlockC")]
    assert ids(scope_boost(chunks, "Year2", ["BlockC"], w_block=0.2, w_year=0.0))[0] == "shared"


def test_zero_weights_keep_the_reranker_order():
    chunks = [page("shared", 20.0), page("y2c", 19.0, "Year2", "BlockC"), page("other", 10.0)]
    assert ids(scope_boost(chunks, "Year2", ["BlockC"], w_block=0.0, w_year=0.0)) == ["shared", "y2c", "other"]


def test_same_block_letter_in_another_year_gets_no_block_bonus():
    chunks = [page("y1c", 20.0, "Year1", "BlockC"), page("y2c", 19.0, "Year2", "BlockC"), page("other", 10.0)]
    assert ids(scope_boost(chunks, "Year2", ["BlockC"], w_block=0.2, w_year=0.0)) == ["y2c", "y1c", "other"]


def test_year_bonus_for_pages_of_the_year():
    chunks = [page("shared", 20.0), page("y2a", 19.0, "Year2", "BlockA"), page("other", 10.0)]
    assert ids(scope_boost(chunks, "Year2", ["BlockC"], w_block=0.0, w_year=0.2))[:2] == ["y2a", "shared"]
    assert ids(scope_boost(chunks, "Year2", None, w_block=0.5, w_year=0.2))[:2] == ["y2a", "shared"]


def test_merged_pages_with_several_blocks_count():
    chunks = [page("shared", 20.0), page("y2ab", 19.0, ["Year2"], ["BlockA", "BlockB"]), page("other", 10.0)]
    assert ids(scope_boost(chunks, "Year2", ["BlockB"], w_block=0.2, w_year=0.0))[:2] == ["y2ab", "shared"]


def test_scores_are_scaled_per_question_so_one_weight_fits_all():
    """ColBERT scores grow with the question's length: the bonus works on min-max scaled scores (0-1)."""
    short = [page("shared", 2.0), page("y2c", 1.9, "Year2", "BlockC"), page("other", 1.0)]
    long = [page("shared", 40.0), page("y2c", 38.0, "Year2", "BlockC"), page("other", 20.0)]
    assert ids(scope_boost(short, "Year2", ["BlockC"], 0.2, 0.0)) == ids(scope_boost(long, "Year2", ["BlockC"], 0.2, 0.0))


def test_no_scope_or_no_chunks_changes_nothing():
    chunks = [page("a", 2.0), page("b", 1.0, "Year2", "BlockC")]
    assert scope_boost(chunks, None, None, 0.5, 0.5) == chunks
    assert scope_boost([], "Year2", ["BlockC"], 0.5, 0.5) == []


def test_equal_scores_do_not_divide_by_zero():
    chunks = [page("shared", 1.0), page("y2c", 1.0, "Year2", "BlockC")]
    assert ids(scope_boost(chunks, "Year2", ["BlockC"], 0.2, 0.0)) == ["y2c", "shared"]


# ---------------------------------------------------------------- in the pipeline


@pytest.fixture
def pipeline(monkeypatch):
    """rag.answer with fake search, rerank and answer; returns the chunks in the order the LLM would see them."""
    found = [page("shared", 20.0), page("y2c", 19.0, "Year2", "BlockC"), page("other", 10.0)]
    monkeypatch.setattr(config, "QUERY_REWRITE", "3")
    monkeypatch.setattr(config, "RERANK_MODEL", "x")
    monkeypatch.setattr(rag, "retrieve_many", lambda *args, **kwargs: list(found))
    monkeypatch.setattr(rag, "rerank", lambda question, chunks: chunks)
    monkeypatch.setattr(rag, "generate", lambda question, chunks, context: "answer")
    monkeypatch.setattr(rag, "describe", lambda student, **kwargs: "")
    monkeypatch.setattr(config, "SCOPE_BOOST_BLOCK", 0.2)
    monkeypatch.setattr(config, "SCOPE_BOOST_YEAR", 0.0)
    student = StudentContext(TODAY, "Year2", "BlockB")
    return lambda question: ids(rag.answer(question, k=3, student=student)["chunks"])


def test_pipeline_boosts_in_boost_mode(pipeline, monkeypatch):
    monkeypatch.setattr(config, "SCOPE_MODE", "boost")
    assert pipeline("what do we have next block?") == ["y2c", "shared", "other"]


def test_pipeline_keeps_the_reranker_order_in_filter_mode(pipeline):
    assert pipeline("what do we have next block?") == ["shared", "y2c", "other"]
