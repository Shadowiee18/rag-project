"""The whole pipeline: question -> retrieve -> generate -> answer + sources.

Run:  uv run python -m ragapp.rag "How do I get VPN access?"
"""

import sys
import time

from ragapp import config
from ragapp.generate import generate
from ragapp.retrieve import retrieve


def answer(question: str, k: int = config.TOP_K, user_year: str | None = None) -> dict:
    started = time.perf_counter()
    chunks = retrieve(question, k, user_year=user_year)
    retrieved_at = time.perf_counter()
    text = generate(question, chunks)
    finished = time.perf_counter()
    return {
        "question": question,
        "answer": text,
        "sources": [
            {"number": n, "doc_id": c["doc_id"], "title": c["title"], "source": c["source"], "score": round(c["score"], 4)}
            for n, c in enumerate(chunks, start=1)
        ],
        "timing_s": {"retrieve": round(retrieved_at - started, 2), "generate": round(finished - retrieved_at, 2)},
    }


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How do I get VPN access?"
    result = answer(question)
    print(f"\nQ: {result['question']}\n\n{result['answer']}\n\nSources:")
    for s in result["sources"]:
        print(f"  [{s['number']}] {s['title']}  ({s['score']})  {s['source']}")
    print(f"\ntiming: {result['timing_s']}")


if __name__ == "__main__":
    main()
