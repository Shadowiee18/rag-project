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


def answer(
    question: str, k: int = config.TOP_K, student: StudentContext | None = None, retrieve_k: int | None = None
) -> dict:
    """k: chunks the LLM sees. retrieve_k: retrieve more than that (the eval uses 10, to see where pages rank)."""
    started = time.perf_counter()
    n = max(k, retrieve_k or 0)
    pool = max(n, config.RERANK_CANDIDATES) if config.RERANK_MODEL else n  # reranking needs candidates to sort
    user_year, user_block, today = (student.year, student.block, student.today) if student else (None, None, None)
    if config.QUERY_REWRITE in ("2", "3"):
        queries = search_queries(question, student) if config.QUERY_REWRITE == "2" else week_queries(question, student)
        rewritten_at = time.perf_counter()
        chunks = retrieve_many(queries, pool, question, user_year, user_block, today)
    else:
        queries = [question]
        if config.QUERY_REWRITE == "1":
            queries = [rewrite_query(question, describe(student, with_plan=False) if student else "")]
        rewritten_at = time.perf_counter()
        chunks = retrieve(queries[0], pool, user_year=user_year, user_block=user_block, today=today)
    search_query = " || ".join(queries)
    retrieved_at = time.perf_counter()
    if config.RERANK_MODEL:
        chunks = rerank(question, chunks)  # the original question: what the student actually asked
    if config.SCOPE_MODE == "boost":  # after reranking: the bonus works on the final relevance scores
        year, blocks = filter_for(question, user_year, user_block, today)
        chunks = scope_boost(chunks, year, blocks, config.SCOPE_BOOST_BLOCK, config.SCOPE_BOOST_YEAR)
    chunks = chunks[:n]
    reranked_at = time.perf_counter()
    text = generate(question, chunks[:k], describe(student) if student else "")
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
