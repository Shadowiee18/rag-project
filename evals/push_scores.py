"""Attach an eval run's grades to its traces in Langfuse, so each answer, its steps and its grades are on one page.

    TRACING=1 uv run python evals/run_eval.py --name my_run     # every result gets the trace id of its question
    (Claude grades the answers: results/my_run_judge_claude.jsonl, see score.py)
    uv run python evals/push_scores.py my_run                   # --dry-run: count the scores, send nothing

Scores per question (Langfuse scores on the question's trace):
  correctness, faithfulness, citations   Claude's grades (categorical), when the judge file exists
  right_page_rank                        1-10: where the first right page ranked (numeric; only when found)
  right_page_in_context                  1/0: a right page was among the chunks the LLM saw (boolean)
The judge's reason goes into the score comment only with TRACE_CONTENT=1 (it describes the answer).
Reads LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY (infra/langfuse/app.env has them).
Score ids are fixed per run, trace and name, so sending a run again on the same day replaces its scores.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
TOP_K = 5  # chunks the LLM sees (config.TOP_K), for right_page_in_context
JUDGE_FIELDS = ("correctness", "faithfulness", "citations")
BATCH = 100


def score_id(run: str, trace_id: str, name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ragapp-eval/{run}/{trace_id}/{name}"))


def build_scores(run: str, results: list[dict], grades: dict[str, dict], with_comments: bool = False) -> list[dict]:
    """The Langfuse score objects for one run. Results without a trace id (run with tracing off) are skipped."""
    scores = []

    def add(trace_id, name, value, data_type, comment=None):
        score = {"id": score_id(run, trace_id, name), "traceId": trace_id, "name": name, "value": value,
                 "dataType": data_type, "metadata": {"run": run}}
        if comment:
            score["comment"] = comment
        scores.append(score)

    for r in results:
        trace_id = r.get("trace_id")
        if not trace_id:
            continue
        grade = grades.get(r["id"], {})
        for name in JUDGE_FIELDS:
            if grade.get(name):
                add(trace_id, name, str(grade[name]), "CATEGORICAL",
                    grade.get("reason") if with_comments and name == "correctness" else None)
        if r.get("source_ids"):  # answerable: retrieval scores
            rank = r.get("hit_rank")
            if rank is not None:
                add(trace_id, "right_page_rank", float(rank), "NUMERIC")
            add(trace_id, "right_page_in_context", 1.0 if rank is not None and rank <= TOP_K else 0.0, "BOOLEAN")
    return scores


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def send(scores: list[dict], host: str, public_key: str, secret_key: str) -> tuple[int, list]:
    """POST the scores in batches. Returns (accepted, errors). A 207 batch is reported, not retried (Langfuse's advice)."""
    import httpx

    accepted, errors = 0, []
    for start in range(0, len(scores), BATCH):
        batch = scores[start: start + BATCH]
        response = httpx.post(f"{host.rstrip('/')}/api/public/scores", json=batch, auth=(public_key, secret_key), timeout=30)
        if response.status_code in (200, 202):
            accepted += len(batch)
        elif response.status_code == 207:
            body = response.json()
            accepted += int(body.get("accepted", 0))  # CreateScoreBatchResults: accepted, rejected, errors
            errors.append(body)
        else:
            raise RuntimeError(f"Langfuse: {response.status_code} {response.text[:300]}")
    return accepted, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run name, e.g. dev_buas27_page")
    parser.add_argument("--dry-run", action="store_true", help="count the scores, send nothing")
    args = parser.parse_args()

    results = read_jsonl(RESULTS_DIR / f"{args.run}.jsonl")
    judge_path = RESULTS_DIR / f"{args.run}_judge_claude.jsonl"
    grades = {g["id"]: g for g in read_jsonl(judge_path)} if judge_path.exists() else {}
    traced = sum(1 for r in results if r.get("trace_id"))
    if not traced:
        sys.exit(f"{args.run}: no result has a trace id; run the eval with TRACING=1")
    scores = build_scores(args.run, results, grades, with_comments=os.getenv("TRACE_CONTENT", "0") == "1")
    print(f"{args.run}: {traced}/{len(results)} questions traced, {len(grades)} graded by Claude -> {len(scores)} scores")
    if args.dry_run:
        return
    accepted, errors = send(scores, os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
                            os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    print(f"accepted {accepted}/{len(scores)}" + (f"; rejected batches: {errors}" if errors else ""))


if __name__ == "__main__":
    main()
