"""The whole pipeline: question (+ who/when) -> rewrite -> retrieve -> generate -> answer + sources.

Run:  uv run python -m ragapp.rag "whats due this week" --year Year2 --block BlockB
"""

import argparse
import time
from datetime import date

from ragapp import config
from ragapp.generate import generate
from ragapp.retrieve import retrieve, retrieve_many
from ragapp.rewrite import rewrite_query, search_queries
from ragapp.student import StudentContext, describe


def answer(
    question: str, k: int = config.TOP_K, student: StudentContext | None = None, retrieve_k: int | None = None
) -> dict:
    """k: chunks the LLM sees. retrieve_k: retrieve more than that (the eval uses 10, to see where pages rank)."""
    started = time.perf_counter()
    n = max(k, retrieve_k or 0)
    user_year, user_block = (student.year, student.block) if student else (None, None)
    if config.QUERY_REWRITE == "2":
        queries = search_queries(question, student)
        rewritten_at = time.perf_counter()
        chunks = retrieve_many(queries, n, question, user_year, user_block)
    else:
        queries = [question]
        if config.QUERY_REWRITE == "1":
            queries = [rewrite_query(question, describe(student, with_plan=False) if student else "")]
        rewritten_at = time.perf_counter()
        chunks = retrieve(queries[0], n, user_year=user_year, user_block=user_block)
    search_query = " || ".join(queries)
    retrieved_at = time.perf_counter()
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
            "generate": round(finished - retrieved_at, 2),
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
