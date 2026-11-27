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
from datetime import date
from pathlib import Path

import ollama

from ragapp import config
from ragapp.cleaning import clean
from ragapp.colbert import load_colbert
from ragapp.rerank import load_model
from ragapp.documents import load_documents
from ragapp.rag import answer as rag_answer
from ragapp.student import StudentContext, describe
from ragapp.tracing import setup as setup_tracing

from groups import group_summary, segment

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_set.jsonl"
RESULTS_DIR = EVALS_DIR / "results"
RETRIEVE_K = 10  # retrieve more than we use, so we can see where the right page ranked
EVAL_TODAY = date(2026, 11, 24)  # fixed, so runs are comparable whenever they happen
DUPLICATE_OF: dict[str, str] = {}  # filled when CLEAN_CORPUS is on: removed copy -> kept page

JUDGE_PROMPT = """You are grading an answer from a study-assistant chatbot.

Question: {question}
Reference answer: {expected}
Chatbot answer: {answer}

Does the chatbot answer contain the key information of the reference answer, without contradicting it?
Reply with exactly one word: CORRECT, PARTIAL or WRONG."""


# 2: whole-word judge labels, half answers judged (see grade). rejudge.py re-grades older saved runs.
JUDGE_RULES = 2
REFUSAL = re.compile(r"\bi (?:don't|do not) know\b")
# Whole words only: "INCORRECT" contains "CORRECT" but means WRONG.
JUDGE_LABEL = re.compile(r"\b(CORRECT|INCORRECT|PARTIAL|PARTIALLY|WRONG)\b")
LABEL_MEANS = {"INCORRECT": "WRONG", "PARTIALLY": "PARTIAL"}


def normalise(answer: str) -> str:
    """Lower-case, with the curly apostrophe LLMs often write turned into a straight one."""
    return answer.lower().replace("’", "'")


def is_refusal(answer: str) -> bool:
    """The answer says "I don't know" somewhere (a full refusal or half an answer)."""
    return bool(REFUSAL.search(normalise(answer)))


def only_refusal(answer: str) -> bool:
    """The answer is "I don't know" and nothing else (citation marks and punctuation aside)."""
    rest = REFUSAL.sub("", re.sub(r"\[\d+\]", "", normalise(answer)))
    return is_refusal(answer) and not re.search(r"\w", rest)


def first_hit_rank(retrieved_doc_ids: list[str], source_ids: list[str]) -> int | None:
    """1-based rank of the first chunk that comes from a correct page, or None."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in source_ids:
            return rank
    return None


def judge(question: str, expected: str, answer: str) -> str:
    response = ollama.chat(
        model=config.JUDGE_MODEL,  # always the local judge, whichever LLM answered
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)}],
        think=False,
        # Same num_ctx as generate.py: a different value makes Ollama reload the model on every call.
        options={"temperature": 0, "num_ctx": 8192},
    )
    match = JUDGE_LABEL.search(response.message.content.upper())
    if not match:
        return "WRONG"
    return LABEL_MEANS.get(match.group(1), match.group(1))


def grade(result: dict) -> str:
    """A bare "I don't know" is wrong without asking; half an answer goes to the judge, which can say PARTIAL."""
    if only_refusal(result["answer"]):
        return "WRONG"
    return judge(result["question"], result["expected_answer"], result["answer"])


def simulated_user_year(item: dict, page_years: dict[str, str]) -> str:
    """Pretend the asker told us their year: the year of the page that answers the question.

    Questions about shared Study Content (no year) or unanswerable ones get 'Year2',
    to check the filter doesn't hurt questions outside the student's own year.
    """
    years = [page_years.get(s, "") for s in item["source_ids"]]
    return next((y for y in years if y), "Year2")


def student_for(item: dict, user_year_mode: str, page_years: dict[str, str]) -> StudentContext | None:
    """A question's own context (the student set's "today 2026-11-24, Year 2, Block B") wins over simulation."""
    ctx = item.get("user_context") or {}
    if ctx.get("year"):
        return StudentContext(today=date.fromisoformat(ctx["today"]), year=ctx["year"], block=ctx.get("block", ""))
    if user_year_mode == "simulated":
        return StudentContext(today=EVAL_TODAY, year=simulated_user_year(item, page_years))
    return None


def evaluate_one(item: dict, student: StudentContext | None = None, judge: str = "claude") -> dict:
    """judge: "claude" = no grading here, Claude grades the saved answers afterwards (score.py);
    "local" = the local LLM judge grades each answer now (used for the Phase 1-2 numbers)."""
    # The same pipeline as the app, but retrieving 10 chunks so we can see where the right page ranks.
    out = rag_answer(item["question"], config.TOP_K, student, retrieve_k=RETRIEVE_K)
    chunks, answer_text = out["chunks"], out["answer"]

    doc_ids = [c["doc_id"] for c in chunks]
    answerable = bool(item["source_ids"])
    result = {
        **item,
        "student": describe(student) if student else None,
        "trace_id": out.get("trace_id"),  # with TRACING=1: push_scores.py attaches the grades to this trace
        "search_query": out["search_query"],
        "answer": answer_text,
        "retrieved_doc_ids": doc_ids,
        # The exact chunks the LLM saw, numbered like in the prompt, for grading faithfulness and citations.
        "context": [{"number": n, "doc_id": c["doc_id"], "text": c["text"]} for n, c in enumerate(chunks[: config.TOP_K], start=1)],
        "refused": is_refusal(answer_text),
        "cites": bool(re.search(r"\[\d+\]", answer_text)),
        "latency_s": out["timing_s"],
    }
    if answerable and item.get("evidence"):
        # Chunk-level check: did the chunks that contain the answer get retrieved?
        phrases = [e["phrase"] for e in item["evidence"]]
        top5_text = "\n".join(c["text"] for c in chunks[: config.TOP_K])
        top10_text = "\n".join(c["text"] for c in chunks)
        result["evidence_found@5"] = sum(p in top5_text for p in phrases) / len(phrases)
        result["evidence_found@10"] = sum(p in top10_text for p in phrases) / len(phrases)
    if answerable:
        # In a cleaned index a merged duplicate's surviving copy counts as that page.
        sources = list(dict.fromkeys(DUPLICATE_OF.get(s, s) for s in item["source_ids"]))
        result["hit_rank"] = first_hit_rank(doc_ids, sources)
        top_used = set(doc_ids[: config.TOP_K])
        result["all_sources_in_context"] = all(s in top_used for s in sources)
        if judge == "local":
            result["judge"] = grade(result)
    return result


def evidence_summary(answerable: list[dict]) -> dict:
    """Chunk-level retrieval: share of answer phrases present in the chunks the LLM saw (top 5) / top 10."""
    with_evidence = [r for r in answerable if "evidence_found@5" in r]
    if not with_evidence:
        return {}
    n = len(with_evidence)
    return {
        "evidence_found@5": round(sum(r["evidence_found@5"] for r in with_evidence) / n, 3),
        "all_evidence_found@5": round(sum(r["evidence_found@5"] == 1 for r in with_evidence) / n, 3),
        "evidence_found@10": round(sum(r["evidence_found@10"] for r in with_evidence) / n, 3),
    }


def answers_summary(results: list[dict]) -> dict:
    """The answer-quality numbers of a run (the part a re-judge recomputes)."""
    answerable = [r for r in results if r["source_ids"]]
    unanswerable = [r for r in results if not r["source_ids"]]
    judged = [r["judge"] for r in answerable if "judge" in r]
    share = lambda label: round(judged.count(label) / len(judged), 3) if judged else None
    return {
        "answers (answerable, LLM-judged)": {
            "graded_by": "local judge" if judged else "Claude, afterwards: evals/score.py <run>",
            "correct": share("CORRECT"),
            "partial": share("PARTIAL"),
            "wrong": share("WRONG"),
            "false_refusals": round(sum(r["refused"] for r in answerable) / len(answerable), 3),
            "cites_sources": round(sum(r["cites"] for r in answerable) / len(answerable), 3),
        },
        "unanswerable_correctly_refused": round(sum(r["refused"] for r in unanswerable) / len(unanswerable), 3) if unanswerable else None,
    }


def summarise(results: list[dict], user_year_mode: str) -> dict:
    answerable = [r for r in results if r["source_ids"]]
    unanswerable = [r for r in results if not r["source_ids"]]
    ranks = [r["hit_rank"] for r in answerable]

    def hit_at(k: int) -> float:
        return round(sum(1 for rank in ranks if rank is not None and rank <= k) / len(answerable), 3)

    generate_times = [r["latency_s"]["generate"] for r in results]
    total_times = [sum(r["latency_s"].values()) for r in results]  # rewrite + retrieve + generate
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
            **evidence_summary(answerable),
        },
        **answers_summary(results),
        "latency_s": {
            "generate_median": round(statistics.median(generate_times), 2),
            "total_median": round(statistics.median(total_times), 2),
            "total_max": round(max(total_times), 2),
        },
        "config": {
            "embed_model": config.EMBED_MODEL, "llm_backend": config.LLM_BACKEND, "student_line_as_source": config.STUDENT_LINE_AS_SOURCE, "llm_model": config.LLM_MODEL,
            "judge_model": config.JUDGE_MODEL,
            "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP, "top_k": config.TOP_K,
            "chunk_headers": config.CHUNK_HEADERS, "filter_by_question_year": config.FILTER_BY_QUESTION_YEAR,
            "scope_mode": config.SCOPE_MODE, "scope_boost_block": config.SCOPE_BOOST_BLOCK, "scope_boost_year": config.SCOPE_BOOST_YEAR,
            "query_rewrite": config.QUERY_REWRITE, "user_year": user_year_mode,
            "collection": config.COLLECTION, "clean_corpus": config.CLEAN_CORPUS, "exclude_templates": config.EXCLUDE_TEMPLATES,
            "rerank_model": config.RERANK_MODEL, "rerank_candidates": config.RERANK_CANDIDATES if config.RERANK_MODEL else None,
            "rerank_max_words": config.RERANK_MAX_WORDS if config.RERANK_MODEL else None,
            "rerank_kind": config.RERANK_KIND if config.RERANK_MODEL else None, "rerank_int8": config.RERANK_INT8,
            "hybrid": config.HYBRID, "judge_rules": JUDGE_RULES,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run", help="name for the results files, e.g. baseline")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--user-year", choices=["none", "simulated"], default="none",
                        help="simulated: pretend each asker told us their year (see simulated_user_year)")
    parser.add_argument("--golden", default=str(GOLDEN_PATH), help="question file, e.g. evals/golden_student.jsonl")
    parser.add_argument("--judge", choices=["claude", "local"], default="claude",
                        help="claude: no grading here, Claude grades the saved answers (score.py). local: the local LLM judge")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev",
                        help="dev: tune and inspect failures here. test: held out, report totals only.")
    args = parser.parse_args()
    setup_tracing()  # TRACING=1: every eval question becomes a trace

    items = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.split != "all":
        items = [item for item in items if item.get("split") == args.split]
    items = items[: args.limit]
    corpus = load_documents(config.CORPUS_PATH)
    page_years = {doc.id: doc.metadata.get("year", "") for doc in corpus}
    page_meta = {doc.id: (doc.metadata.get("year", ""), doc.metadata.get("block", "")) for doc in corpus}
    for item in items:
        item["segment"] = segment(item["source_ids"], page_meta)  # where the answer lives (see groups.py)
    if config.CLEAN_CORPUS:
        DUPLICATE_OF.update(clean(corpus).duplicate_of)

    RESULTS_DIR.mkdir(exist_ok=True)
    if config.RERANK_MODEL:  # load once before timing, so question 1 isn't charged for it
        load_colbert() if config.RERANK_KIND == "colbert" else load_model(config.RERANK_MODEL)
    results = []
    for number, item in enumerate(items, start=1):
        results.append(evaluate_one(item, student_for(item, args.user_year, page_years), args.judge))
        print(f"  {number}/{len(items)}  {item['id']}", end="\r")
    print()

    with open(RESULTS_DIR / f"{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(results, args.user_year)
    summary["config"]["split"] = args.split
    summary["config"]["judge"] = args.judge
    summary["config"]["golden"] = Path(args.golden).name
    if args.split == "dev":  # per-group numbers only where failures may be inspected; test stays totals only
        summary["by_question_type"] = group_summary(results, "question_type")
        summary["by_segment"] = group_summary(results, "segment")
    (RESULTS_DIR / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
