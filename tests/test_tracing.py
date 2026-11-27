"""Tracing: every step of a question becomes an OpenTelemetry span (viewable in Langfuse or any OTLP backend).

Spans are collected in memory here: no Langfuse, Qdrant or Ollama needed.
"""

import json
from datetime import date

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import ragapp.rag as rag
import ragapp.tracing as tracing
from ragapp import config
from ragapp.student import StudentContext

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)  # once per process: OpenTelemetry allows one global provider

QUESTION = "what do we have next block?"
STUDENT = StudentContext(date(2026, 11, 26), "Year2", "BlockB")


def chunk(doc_id, score, year="", block=""):
    return {"doc_id": doc_id, "chunk_index": 0, "score": score, "year": year, "block": block, "title": doc_id,
            "source": "", "text": "course text " + doc_id, "point_id": doc_id}


@pytest.fixture
def pipeline(monkeypatch):
    """rag.answer with fake search, rerank and LLM."""
    EXPORTER.clear()
    found = [chunk("General/Rules", 20.0), chunk("Year2/BlockC", 19.0, "Year2", "BlockC")]
    monkeypatch.setattr(config, "QUERY_REWRITE", "3")
    monkeypatch.setattr(config, "RERANK_MODEL", "x")
    monkeypatch.setattr(config, "TRACE_CONTENT", False)
    monkeypatch.setattr(rag, "retrieve_many", lambda *args, **kwargs: list(found))
    monkeypatch.setattr(rag, "rerank", lambda question, chunks: list(reversed(chunks)))
    monkeypatch.setattr(rag, "generate", lambda question, chunks, context: "Block C covers deep learning [1].")
    monkeypatch.setattr(rag, "describe", lambda student, **kwargs: "Today is Thursday.")
    return lambda: rag.answer(QUESTION, k=2, student=STUDENT)


def spans() -> dict:
    return {s.name: s for s in EXPORTER.get_finished_spans()}


def all_attribute_text() -> str:
    return json.dumps([dict(s.attributes) for s in EXPORTER.get_finished_spans()])


def test_one_span_per_step_under_one_trace(pipeline):
    pipeline()
    found = spans()
    assert set(found) == {"answer", "rewrite", "retrieve", "rerank", "generate"}
    root = found["answer"]
    assert root.parent is None
    for name in ("rewrite", "retrieve", "rerank", "generate"):
        assert found[name].parent.span_id == root.context.span_id
        assert found[name].context.trace_id == root.context.trace_id


def test_retrieval_spans_show_pages_scores_and_scope(pipeline):
    pipeline()
    retrieve, rerank = spans()["retrieve"], spans()["rerank"]
    assert retrieve.attributes["langfuse.observation.type"] == "retriever"
    assert json.loads(retrieve.attributes["langfuse.observation.output"])[0] == {"doc_id": "General/Rules", "score": 20.0}
    assert retrieve.attributes["langfuse.observation.metadata.scope"] == "Year2 BlockC"
    assert retrieve.attributes["langfuse.observation.metadata.candidates"] == 2
    # the reranker's new order is visible: the block page moved up
    assert json.loads(rerank.attributes["langfuse.observation.output"])[0]["doc_id"] == "Year2/BlockC"


def test_generation_span_names_the_model(pipeline):
    pipeline()
    generate = spans()["generate"]
    assert generate.attributes["langfuse.observation.type"] == "generation"
    assert generate.attributes["gen_ai.request.model"] == config.LLM_MODEL
    assert json.loads(generate.attributes["langfuse.observation.metadata.context_pages"]) == ["Year2/BlockC", "General/Rules"]


def test_trace_carries_the_students_scope_not_who_they_are(pipeline):
    pipeline()
    root = spans()["answer"]
    assert root.attributes["langfuse.trace.name"] == "answer"
    assert root.attributes["langfuse.trace.metadata.year"] == "Year2"
    assert root.attributes["langfuse.trace.metadata.block"] == "BlockB"


def test_no_question_answer_or_course_text_by_default(pipeline):
    """Questions and answers are user content: recorded only with TRACE_CONTENT=1 (the OpenTelemetry default)."""
    pipeline()
    text = all_attribute_text()
    assert QUESTION not in text and "deep learning" not in text and "course text" not in text
    assert "Today is Thursday" not in text


def test_question_and_answer_with_trace_content_on(pipeline, monkeypatch):
    monkeypatch.setattr(config, "TRACE_CONTENT", True)
    pipeline()
    root, rewrite = spans()["answer"], spans()["rewrite"]
    assert json.loads(root.attributes["langfuse.observation.input"]) == QUESTION
    assert json.loads(root.attributes["langfuse.observation.output"]) == "Block C covers deep learning [1]."
    assert QUESTION in json.loads(rewrite.attributes["langfuse.observation.output"])
    assert "course text" not in all_attribute_text()  # chunk texts never: page ids and scores are enough


def test_a_failing_step_is_marked_as_an_error(pipeline, monkeypatch):
    def broken(question, chunks, context):
        raise RuntimeError("BUas LLM server: 503")

    monkeypatch.setattr(rag, "generate", broken)
    with pytest.raises(RuntimeError):
        pipeline()
    assert spans()["generate"].status.status_code == StatusCode.ERROR
    assert spans()["answer"].status.status_code == StatusCode.ERROR


def test_boost_mode_shows_its_own_step(pipeline, monkeypatch):
    monkeypatch.setattr(config, "SCOPE_MODE", "boost")
    pipeline()
    assert "scope_boost" in spans()


# ---------------------------------------------------------------- setup


def test_setup_does_nothing_with_tracing_off(monkeypatch):
    monkeypatch.setattr(config, "TRACING", False)
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", lambda p: pytest.fail("must not install a provider"))
    assert tracing.setup() is False


def test_setup_sends_spans_over_otlp_with_tracing_on(monkeypatch):
    installed = []
    monkeypatch.setattr(config, "TRACING", True)
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", installed.append)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:3000/api/public/otel")
    assert tracing.setup() is True
    (provider,) = installed
    assert provider.resource.attributes["service.name"] == "ragapp"
