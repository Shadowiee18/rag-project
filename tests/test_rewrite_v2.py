from datetime import date

import ragapp.rewrite as rewrite
from ragapp.retrieve import rrf_merge
from ragapp.student import StudentContext


def chunk(doc, i):
    return {"doc_id": doc, "chunk_index": i, "text": f"{doc}#{i}", "score": 0.5}


def test_rrf_prefers_chunks_found_by_both_queries():
    a = [chunk("X", 0), chunk("Y", 0), chunk("Z", 0)]
    b = [chunk("Y", 0), chunk("W", 0), chunk("X", 0)]
    merged = rrf_merge([a, b], k=4)
    assert [c["doc_id"] for c in merged[:2]] == ["Y", "X"]  # in both lists
    assert {c["doc_id"] for c in merged} == {"X", "Y", "Z", "W"}


def test_rrf_respects_k_and_keeps_one_copy_per_chunk():
    merged = rrf_merge([[chunk("X", 0), chunk("X", 1)], [chunk("X", 0)]], k=5)
    assert [(c["doc_id"], c["chunk_index"]) for c in merged] == [("X", 0), ("X", 1)]


def test_expand_strips_year_block_week_the_model_adds(monkeypatch):
    monkeypatch.setattr(rewrite.llm, "chat", lambda messages: "Year 2 Block B week 3 Keras backend TensorFlow")
    assert rewrite.expand_query("keras backend") == "Keras backend TensorFlow"


def test_expand_keeps_year_block_the_question_itself_names(monkeypatch):
    monkeypatch.setattr(rewrite.llm, "chat", lambda messages: "Year 2 Block A data quality report deadline")
    assert "Block A" in rewrite.expand_query("deadline data quality report year 2 block a")


def test_code_adds_student_week_only_for_time_questions(monkeypatch):
    monkeypatch.setattr(rewrite, "expand_query", lambda q: q + " expanded")
    monkeypatch.setattr(rewrite, "search_phrase", lambda s: "Year 2 Block B week 3")
    student = StudentContext(today=date(2026, 11, 24), year="Year2", block="BlockB")

    assert rewrite.search_queries("whats due this week", student) == [
        "Year 2 Block B week 3 whats due this week",
        "Year 2 Block B week 3 whats due this week expanded",
    ]
    assert rewrite.search_queries("keras backend", student) == ["keras backend", "keras backend expanded"]
    # names its own block: no student context added
    assert rewrite.search_queries("year 2 block a deadline", student)[0] == "year 2 block a deadline"


def test_week_queries_add_a_week_tagged_copy_only_for_time_questions(monkeypatch):
    monkeypatch.setattr(rewrite, "search_phrase", lambda ctx: "Year 2 Block B week 3")
    student = StudentContext(today=date(2026, 11, 24), year="Year2", block="BlockB")
    assert rewrite.week_queries("whats due this week", student) == [
        "whats due this week",
        "Year 2 Block B week 3 whats due this week",
    ]
    assert rewrite.week_queries("keras backend", student) == ["keras backend"]  # not about time
    assert rewrite.week_queries("deadline year 1 block a", student) == ["deadline year 1 block a"]  # names its own block
    assert rewrite.week_queries("whats due this week", None) == ["whats due this week"]  # nobody known
