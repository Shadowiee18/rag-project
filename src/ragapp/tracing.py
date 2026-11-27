"""Tracing: every step of every question as an OpenTelemetry span, so a bad answer can be looked up instead of re-run.

    answer                       question in, answer out; the asker's year/block
    ├── rewrite                  the search queries
    ├── retrieve   (retriever)   candidates: page ids + scores, and the scope searched
    ├── rerank     (retriever)   the new order
    ├── scope_boost              only with SCOPE_MODE=boost
    ├── calendar                 only with CALENDAR_FACTS=1: the calendar facts given to the model
    └── generate   (generation)  model, the pages it saw, the answer

Plain OpenTelemetry, so any OTLP backend can show it. The attribute names Langfuse reads (langfuse.observation.*,
langfuse.trace.*) are ordinary span attributes; other backends ignore them.

Off unless TRACING=1. The exporter uses the standard OpenTelemetry environment variables, e.g. for a local Langfuse:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000/api/public/otel
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64 of public_key:secret_key>
Questions, queries and answers are user content: only recorded with TRACE_CONTENT=1. Chunk texts never are.
"""

import json

from opentelemetry import trace
from opentelemetry.trace import Span

from ragapp import config

tracer = trace.get_tracer("ragapp")


def setup() -> bool:
    """Send spans to the OTLP endpoint (TRACING=1). Call once at startup. Returns whether tracing is on."""
    if not config.TRACING:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "ragapp"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # sends in the background, in batches
    trace.set_tracer_provider(provider)
    return True


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def record(span: Span, kind: str | None = None, input=None, output=None, content: bool = False, **metadata) -> None:
    """Set a step's type, input, output and metadata. content=True: input/output are user content, only recorded
    with TRACE_CONTENT=1."""
    if kind:
        span.set_attribute("langfuse.observation.type", kind)
    if not content or config.TRACE_CONTENT:
        if input is not None:
            span.set_attribute("langfuse.observation.input", _json(input))
        if output is not None:
            span.set_attribute("langfuse.observation.output", _json(output))
    for key, value in metadata.items():
        if value is not None:
            span.set_attribute(f"langfuse.observation.metadata.{key}",
                               value if isinstance(value, (str, int, float, bool)) else _json(value))


def pages(chunks: list[dict], limit: int = 10) -> list[dict]:
    """What a retrieval step found, without the course text: page id and score, best first."""
    return [{"doc_id": c["doc_id"], "score": round(float(c["score"]), 4)} for c in chunks[:limit]]
