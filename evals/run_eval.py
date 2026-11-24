"""Score the RAG system against evals/golden_set.jsonl.

Run:  uv run python evals/run_eval.py --name baseline
      uv run python evals/run_eval.py --name quick --limit 5

Writes evals/results/<name>.jsonl (every question, answer and score)
and   evals/results/<name>_summary.json (the headline numbers).
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import ollama

from ragapp import config
from ragapp.documents import load_documents
from ragapp.generate import generate
from ragapp.retrieve import retrieve

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_set.jsonl"
RESULTS_DIR = EVALS_DIR / "results"
RETRIEVE_K = 10  # retrieve more than we use, so we can see where the right page ranked

JUDGE_PROMPT = """You are grading an answer from a study-assistant chatbot.

Question: {question}
Reference answer: {expected}
Chatbot answer: {answer}

Does the chatbot answer contain the key information of the reference answer, without contradicting it?
Reply with exactly one word: CORRECT, PARTIAL or WRONG."""


def is_refusal(answer: str) -> bool:
    return "i don't know" in answer.lower() or "i do not know" in answer.lower()


def first_hit_rank(retrieved_doc_ids: list[str], source_ids: list[str]) -> int | None:
    """1-based rank of the first chunk that comes from a correct page, or None."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in source_ids:
            return rank
    return None


def judge(question: str, expected: str, answer: str) -> str:
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)}],
        think=False,
        # Same num_ctx as generate.py: a different value makes Ollama reload the model on every call.
        options={"temperature": 0, "num_ctx": 8192},
    )
    word = response.message.content.strip().upper()
    for label in ("CORRECT", "PARTIAL", "WRONG"):
        if label in word:
            return label
    return "WRONG"


def simulated_user_year(item: dict, page_years: dict[str, str]) -> str:
    """Pretend the asker told us their year: the year of the page that answers the question.

    Questions about shared Study Content (no year) or unanswerable ones get 'Year2',
    to check the filter doesn't hurt questions outside the student's own year.
    """
    years = [page_years.get(s, "") for s in item["source_ids"]]
    return next((y for y in years if y), "Year2")


def evaluate_one(item: dict, user_year: str | None = None) -> dict:
    started = time.perf_counter()
    chunks = retrieve(item["question"], k=RETRIEVE_K, user_year=user_year)
    retrieved_at = time.perf_counter()
    answer = generate(item["question"], chunks[: config.TOP_K])
    finished = time.perf_counter()

    doc_ids = [c["doc_id"] for c in chunks]
    answerable = bool(item["source_ids"])
    result = {
        **item,
        "user_year": user_year,
        "answer": answer,
        "retrieved_doc_ids": doc_ids,
        "refused": is_refusal(answer),
        "cites": bool(re.search(r"\[\d+\]", answer)),
        "latency_s": {"retrieve": round(retrieved_at - started, 2), "generate": round(finished - retrieved_at, 2)},
    }
    if answerable:
        result["hit_rank"] = first_hit_rank(doc_ids, item["source_ids"])
        top_used = set(doc_ids[: config.TOP_K])
        result["all_sources_in_context"] = all(s in top_used for s in item["source_ids"])
        result["judge"] = "WRONG" if result["refused"] else judge(item["question"], item["expected_answer"], answer)
    return result


def summarise(results: list[dict], user_year_mode: str) -> dict:
    answerable = [r for r in results if r["source_ids"]]
    unanswerable = [r for r in results if not r["source_ids"]]
    ranks = [r["hit_rank"] for r in answerable]

    def hit_at(k: int) -> float:
        return round(sum(1 for rank in ranks if rank is not None and rank <= k) / len(answerable), 3)

    judged = [r["judge"] for r in answerable]
    generate_times = [r["latency_s"]["generate"] for r in results]
    total_times = [r["latency_s"]["retrieve"] + r["latency_s"]["generate"] for r in results]
    return {
        "questions": len(results),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval": {
            "hit@1": hit_at(1),
            "hit@3": hit_at(3),
            f"hit@{config.TOP_K} (what the LLM sees)": hit_at(config.TOP_K),
            "hit@10": hit_at(10),
            "mrr@10": round(sum(1 / rank for rank in ranks if rank) / len(answerable), 3),
            "all_sources_in_context": round(sum(r["all_sources_in_context"] for r in answerable) / len(answerable), 3),
        },
        "answers (answerable, LLM-judged)": {
            "correct": round(judged.count("CORRECT") / len(answerable), 3),
            "partial": round(judged.count("PARTIAL") / len(answerable), 3),
            "wrong": round(judged.count("WRONG") / len(answerable), 3),
            "false_refusals": round(sum(r["refused"] for r in answerable) / len(answerable), 3),
            "cites_sources": round(sum(r["cites"] for r in answerable) / len(answerable), 3),
        },
        "unanswerable_correctly_refused": round(sum(r["refused"] for r in unanswerable) / len(unanswerable), 3) if unanswerable else None,
        "latency_s": {
            "generate_median": round(statistics.median(generate_times), 2),
            "total_median": round(statistics.median(total_times), 2),
            "total_max": round(max(total_times), 2),
        },
        "config": {
            "embed_model": config.EMBED_MODEL, "llm_model": config.LLM_MODEL,
            "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP, "top_k": config.TOP_K,
            "chunk_headers": config.CHUNK_HEADERS, "filter_by_question_year": config.FILTER_BY_QUESTION_YEAR,
            "user_year": user_year_mode,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run", help="name for the results files, e.g. baseline")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--user-year", choices=["none", "simulated"], default="none",
                        help="simulated: pretend each asker told us their year (see simulated_user_year)")
    args = parser.parse_args()

    items = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = items[: args.limit]
    page_years = {doc.id: doc.metadata.get("year", "") for doc in load_documents(config.CORPUS_PATH)}

    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for number, item in enumerate(items, start=1):
        user_year = simulated_user_year(item, page_years) if args.user_year == "simulated" else None
        results.append(evaluate_one(item, user_year))
        print(f"  {number}/{len(items)}  {item['id']}", end="\r")
    print()

    with open(RESULTS_DIR / f"{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(results, args.user_year)
    (RESULTS_DIR / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
