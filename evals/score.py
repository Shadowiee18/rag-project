"""Score a finished eval run with extra judges, without re-running the RAG system.

Run:  uv run python evals/score.py phase2_start

Reads   evals/results/<run>.jsonl                 (answers + the chunks each answer saw)
        evals/golden_set.jsonl                    (evidence phrases, if present)
        evals/results/<run>_judge_claude.jsonl    (optional: grades by Claude, following judge_rubric.md)
        evals/results/<run>_judge_human.jsonl     (optional: grades by a person, same format)
Writes  evals/results/<run>_score.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"
LOCAL_TO_RUBRIC = {"CORRECT": "correct", "PARTIAL": "partial", "WRONG": "wrong"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Agreement between two graders, corrected for the agreement you'd get by chance.

    1 = perfect, 0 = no better than chance. Rules of thumb: >0.6 acceptable, >0.8 strong.
    """
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(count_a[label] * count_b[label] for label in set(a) | set(b)) / (n * n)
    return round((observed - expected) / (1 - expected), 3) if expected < 1 else 1.0


def judge_stats(grades: dict[str, dict], results: dict[str, dict]) -> dict:
    answerable = [g for qid, g in grades.items() if results[qid]["source_ids"]]
    unanswerable = [g for qid, g in grades.items() if not results[qid]["source_ids"]]
    share = lambda items, key, value: round(sum(g[key] == value for g in items) / len(items), 3) if items else None
    return {
        "graded": len(grades),
        "correct": share(answerable, "correctness", "correct"),
        "partial": share(answerable, "correctness", "partial"),
        "wrong": share(answerable, "correctness", "wrong"),
        "faithful (all)": share(list(grades.values()), "faithfulness", "faithful"),
        "citations ok (answered)": share([g for g in grades.values() if g["citations"] != "n/a"], "citations", "ok"),
        "unanswerable correctly refused": share(unanswerable, "correctness", "correct"),
    }


def agreement(name_a: str, a: dict[str, str], name_b: str, b: dict[str, str]) -> dict:
    shared = sorted(set(a) & set(b))
    labels_a, labels_b = [a[q] for q in shared], [b[q] for q in shared]
    return {
        "pair": f"{name_a} vs {name_b}",
        "questions": len(shared),
        "same_label": round(sum(x == y for x, y in zip(labels_a, labels_b)) / len(shared), 3),
        "cohens_kappa": cohens_kappa(labels_a, labels_b),
        "disagreements": [f"{q}: {a[q]} vs {b[q]}" for q in shared if a[q] != b[q]],
    }


def main() -> None:
    run = sys.argv[1]
    results = {r["id"]: r for r in read_jsonl(RESULTS_DIR / f"{run}.jsonl")}
    golden = {g["id"]: g for g in read_jsonl(EVALS_DIR / "golden_set.jsonl")}
    score: dict = {"run": run}

    # Chunk-level retrieval from the saved context (top 5 = what the LLM saw).
    found = []
    for qid, r in results.items():
        evidence = golden.get(qid, {}).get("evidence")
        if r["source_ids"] and evidence and "context" in r:
            context = "\n".join(c["text"] for c in r["context"])
            found.append(sum(e["phrase"] in context for e in evidence) / len(evidence))
    if found:
        score["evidence_found@5"] = round(sum(found) / len(found), 3)
        score["all_evidence_found@5"] = round(sum(f == 1 for f in found) / len(found), 3)

    # Judges. The local judge only grades correctness of answerable questions.
    labels = {"local": {q: LOCAL_TO_RUBRIC[r["judge"]] for q, r in results.items() if "judge" in r}}
    for judge in ("claude", "human"):
        path = RESULTS_DIR / f"{run}_judge_{judge}.jsonl"
        if path.exists():
            grades = {g["id"]: g for g in read_jsonl(path)}
            score[f"judge_{judge}"] = judge_stats(grades, results)
            labels[judge] = {q: g["correctness"] for q, g in grades.items() if results[q]["source_ids"]}

    names = list(labels)
    score["agreement (correctness, answerable)"] = [
        agreement(names[i], labels[names[i]], names[j], labels[names[j]])
        for i in range(len(names)) for j in range(i + 1, len(names))
    ]

    (RESULTS_DIR / f"{run}_score.json").write_text(json.dumps(score, indent=2), encoding="utf-8")
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
