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


def test_generate_step_shows_the_pages_the_model_saw(pipeline):
    pipeline()
    generate = spans()["generate"]
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


# ---------------------------------------------------------------- every LLM call: model and tokens

import ragapp.llm as llm  # noqa: E402


class _Reply:
    """A fake Ollama reply with its token counts."""

    def __init__(self, text, prompt_eval_count=None, eval_count=None):
        self.message = type("m", (), {"content": text})
        self.prompt_eval_count, self.eval_count = prompt_eval_count, eval_count


class _Http:
    def __init__(self, body):
        self.status_code, self._body, self.text = 200, body, str(body)

    def json(self):
        return self._body


def llm_span():
    (span,) = [s for s in EXPORTER.get_finished_spans() if s.name == "llm"]
    return span


@pytest.fixture
def no_content(monkeypatch):
    EXPORTER.clear()
    monkeypatch.setattr(config, "TRACE_CONTENT", False)


def test_ollama_call_is_a_generation_with_model_and_tokens(no_content, monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(config, "LLM_MODEL", "qwen3:8b")
    monkeypatch.setattr(llm.ollama, "chat", lambda **kw: _Reply("answer", prompt_eval_count=812, eval_count=64))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "answer"
    span = llm_span()
    assert span.attributes["langfuse.observation.type"] == "generation"
    assert span.attributes["langfuse.observation.model.name"] == "qwen3:8b"
    assert json.loads(span.attributes["langfuse.observation.usage_details"]) == {"input": 812, "output": 64}
    assert span.attributes["gen_ai.usage.input_tokens"] == 812 and span.attributes["gen_ai.usage.output_tokens"] == 64


def test_buas_call_records_the_usage_the_api_reports(no_content, monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "buas")
    monkeypatch.setattr(config, "LLM_MODEL", "qwen3.8-27b")
    monkeypatch.setattr(config, "BUAS_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setattr(config, "BUAS_LLM_API_KEY", "test-key")
    body = {"choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 120, "total_tokens": 1620}}
    monkeypatch.setattr(llm.httpx, "post", lambda url, **kw: _Http(body))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "answer"
    span = llm_span()
    assert span.attributes["langfuse.observation.model.name"] == "qwen3.8-27b"
    assert span.attributes["langfuse.observation.metadata.backend"] == "buas"
    assert json.loads(span.attributes["langfuse.observation.usage_details"]) == {"input": 1500, "output": 120}


def test_no_usage_reported_means_no_usage_recorded(no_content, monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(llm.ollama, "chat", lambda **kw: _Reply("answer"))
    llm.chat([{"role": "user", "content": "hi"}])
    assert "langfuse.observation.usage_details" not in llm_span().attributes


def test_llm_prompt_and_reply_are_content(no_content, monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(llm.ollama, "chat", lambda **kw: _Reply("the secret answer", 10, 2))
    llm.chat([{"role": "user", "content": "my private question"}])
    assert "private question" not in json.dumps(dict(llm_span().attributes))
    assert "secret answer" not in json.dumps(dict(llm_span().attributes))
    EXPORTER.clear()
    monkeypatch.setattr(config, "TRACE_CONTENT", True)
    llm.chat([{"role": "user", "content": "my private question"}])
    attrs = llm_span().attributes
    assert json.loads(attrs["langfuse.observation.input"]) == [{"role": "user", "content": "my private question"}]
    assert json.loads(attrs["langfuse.observation.output"]) == "the secret answer"


def test_answer_trace_has_an_llm_call_under_generate(monkeypatch):
    """With the real generate(): the answer's LLM call is a child of the generate step."""
    EXPORTER.clear()
    import ragapp.generate as generate_module

    monkeypatch.setattr(config, "QUERY_REWRITE", "3")
    monkeypatch.setattr(config, "RERANK_MODEL", "")
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(rag, "retrieve_many", lambda *a, **k: [chunk("General/Rules", 1.0)])
    monkeypatch.setattr(rag, "describe", lambda student, **kwargs: "")
    monkeypatch.setattr(rag, "generate", generate_module.generate)
    monkeypatch.setattr(llm.ollama, "chat", lambda **kw: _Reply("answer [1]", 300, 20))
    rag.answer(QUESTION, k=1, student=STUDENT)
    found = spans()
    assert found["llm"].parent.span_id == found["generate"].context.span_id
    assert found["generate"].attributes.get("langfuse.observation.type") != "generation"  # one generation per call


def test_answer_result_carries_its_trace_id(pipeline):
    """So an eval run can attach its grades to the trace afterwards."""
    result = pipeline()
    assert result["trace_id"] == format(spans()["answer"].context.trace_id, "032x")
