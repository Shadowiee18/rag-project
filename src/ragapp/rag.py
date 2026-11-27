"""The whole pipeline: question (+ who/when) -> rewrite -> retrieve -> generate -> answer + sources.

Run:  uv run python -m ragapp.rag "whats due this week" --year Year2 --block BlockB
"""

import argparse
import time
from datetime import date

from ragapp import config
from ragapp.generate import generate
from ragapp.rerank import rerank
from ragapp.retrieve import filter_for, retrieve, retrieve_many, scope_boost
from ragapp.rewrite import rewrite_query, search_queries, week_queries
from ragapp.student import StudentContext, describe
from ragapp.tracing import pages, record, setup as setup_tracing, tracer


def answer(
    question: str, k: int = config.TOP_K, student: StudentContext | None = None, retrieve_k: int | None = None
) -> dict:
    """k: chunks the LLM sees. retrieve_k: retrieve more than that (the eval uses 10, to see where pages rank)."""
    started = time.perf_counter()
    n = max(k, retrieve_k or 0)
    pool = max(n, config.RERANK_CANDIDATES) if config.RERANK_MODEL else n  # reranking needs candidates to sort
    user_year, user_block, today = (student.year, student.block, student.today) if student else (None, None, None)
    with tracer.start_as_current_span("answer") as root:
        root.set_attribute("langfuse.trace.name", "answer")
        for key, value in (("year", user_year), ("block", user_block), ("today", today and str(today))):
            if value:
                root.set_attribute(f"langfuse.trace.metadata.{key}", value)
        record(root, input=question, content=True)

        with tracer.start_as_current_span("rewrite") as span:
            if config.QUERY_REWRITE in ("2", "3"):
                queries = search_queries(question, student) if config.QUERY_REWRITE == "2" else week_queries(question, student)
            else:
                queries = [question]
                if config.QUERY_REWRITE == "1":
                    queries = [rewrite_query(question, describe(student, with_plan=False) if student else "")]
            record(span, output=queries, content=True, mode=config.QUERY_REWRITE, queries=len(queries))
        rewritten_at = time.perf_counter()

        year, blocks = filter_for(question, user_year, user_block, today)
        with tracer.start_as_current_span("retrieve") as span:
            if config.QUERY_REWRITE in ("2", "3"):
                chunks = retrieve_many(queries, pool, question, user_year, user_block, today)
            else:
                chunks = retrieve(queries[0], pool, user_year=user_year, user_block=user_block, today=today)
            record(span, "retriever", output=pages(chunks), candidates=len(chunks), hybrid=config.HYBRID,
                   scope=" ".join([year or "", *(blocks or [])]).strip() or "all", scope_mode=config.SCOPE_MODE)
        search_query = " || ".join(queries)
        retrieved_at = time.perf_counter()

        if config.RERANK_MODEL:
            with tracer.start_as_current_span("rerank") as span:
                chunks = rerank(question, chunks)  # the original question: what the student actually asked
                record(span, "retriever", output=pages(chunks), reranker=config.RERANK_KIND)
        if config.SCOPE_MODE == "boost":  # after reranking: the bonus works on the final relevance scores
            with tracer.start_as_current_span("scope_boost") as span:
                chunks = scope_boost(chunks, year, blocks, config.SCOPE_BOOST_BLOCK, config.SCOPE_BOOST_YEAR)
                record(span, output=pages(chunks), w_block=config.SCOPE_BOOST_BLOCK, w_year=config.SCOPE_BOOST_YEAR)
        chunks = chunks[:n]
        reranked_at = time.perf_counter()

        with tracer.start_as_current_span("generate") as span:
            span.set_attribute("gen_ai.request.model", config.LLM_MODEL)
            text = generate(question, chunks[:k], describe(student) if student else "")
            record(span, "generation", output=text, content=True, backend=config.LLM_BACKEND,
                   context_pages=[c["doc_id"] for c in chunks[:k]])
        record(root, output=text, content=True)
    finished = time.perf_counter()
    return {
        "question": question,
        "search_query": search_query,
        "answer": text,
        "chunks": chunks,
        "sources": [
            {"number": n, "doc_id": c["doc_id"], "title": c["title"], "source": c["source"], "score": round(c["score"], 4)}
            for n, c in enumerate(chunks[:k], start=1)
        ],
        "timing_s": {
            "rewrite": round(rewritten_at - started, 2),
            "retrieve": round(retrieved_at - rewritten_at, 2),
            "rerank": round(reranked_at - retrieved_at, 2),
            "generate": round(finished - reranked_at, 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    parser.add_argument("--year", default="", help="e.g. Year2")
    parser.add_argument("--block", default="", help="e.g. BlockB")
    parser.add_argument("--today", default="", help="YYYY-MM-DD, default: today")
    args = parser.parse_args()
    setup_tracing()

    student = None
    if args.year or args.today:
        today = date.fromisoformat(args.today) if args.today else date.today()
        student = StudentContext(today=today, year=args.year, block=args.block)

    result = answer(" ".join(args.question), student=student)
    print(f"\nQ: {result['question']}")
    if result["search_query"] != result["question"]:
        print(f"   (searched for: {result['search_query']})")
    print(f"\n{result['answer']}\n\nSources:")
    for s in result["sources"]:
        print(f"  [{s['number']}] {s['title']}  ({s['score']})  {s['source']}")
    print(f"\ntiming: {result['timing_s']}")


if __name__ == "__main__":
    main()
