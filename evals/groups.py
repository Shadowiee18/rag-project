"""Eval results per group of questions, so an average can't hide where the system fails.

Two labels per question:
- question_type (in the question files; labelled by an LLM from the question text only, not reviewed by a person;
  step 9's router labels them independently, and disagreements are what gets reviewed):
    schedule | deadline | overview | rules | howto | concept | people | out_of_scope
  A question asking two things gets the riskier label: deadline > rules > the rest.
- segment (computed here): where the answer lives, from its source pages' year/block metadata.

Run:  uv run python evals/groups.py                                      coverage: questions per type x segment
      uv run python evals/groups.py --run dev_hyb_colbert_c30_page        a saved dev run, per group
      (add --golden evals/golden_student.jsonl for the student-style set)
"""

import argparse
import json
from pathlib import Path

from ragapp import config
from ragapp.documents import load_documents

EVALS_DIR = Path(__file__).resolve().parent
QUESTION_TYPES = ("schedule", "deadline", "overview", "rules", "howto", "concept", "people", "out_of_scope")
NON_SITE_SEGMENTS = ("shared", "several", "none")


def page_label(year: str, block: str) -> str:
    if not year:
        return "shared"  # Study Content and other pages that belong to no year
    return f"{year} {block}" if block else year


def segment(source_ids: list[str], page_meta: dict[str, tuple[str, str]]) -> str:
    """'Year2 BlockB', 'Year3', 'shared', 'several' (answer spread over segments) or 'none' (unanswerable)."""
    if not source_ids:
        return "none"
    labels = {page_label(*page_meta[s]) for s in source_ids}
    return labels.pop() if len(labels) == 1 else "several"


def share(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 3)


def numbers(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["source_ids"]]
    unanswerable = [r for r in rows if not r["source_ids"]]
    out: dict = {"questions": len(rows)}
    if answerable:
        out["hit@5"] = share([r["hit_rank"] is not None and r["hit_rank"] <= config.TOP_K for r in answerable])
        with_evidence = [r for r in answerable if "evidence_found@5" in r]
        if with_evidence:
            out["all_evidence@5"] = share([r["evidence_found@5"] == 1 for r in with_evidence])
        out["correct"] = share([r["judge"] == "CORRECT" for r in answerable])
    if unanswerable:
        out["refused"] = share([r["refused"] for r in unanswerable])
    return out


def group_summary(results: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r.get(key) or "untagged", []).append(r)
    return {name: numbers(rows) for name, rows in sorted(groups.items())}


def site_segments(page_meta: dict[str, tuple[str, str]]) -> list[str]:
    on_site = sorted({page_label(*meta) for meta in page_meta.values()} - set(NON_SITE_SEGMENTS))
    return on_site + list(NON_SITE_SEGMENTS)


def coverage(items: list[dict], page_meta: dict[str, tuple[str, str]]) -> dict:
    """{question_type: {segment: {"dev": n, "test": n}}} with every type and every segment, empty cells included."""
    table = {t: {s: {"dev": 0, "test": 0} for s in site_segments(page_meta)} for t in QUESTION_TYPES}
    for item in items:
        table[item["question_type"]][item["segment"]][item["split"]] += 1
    return table


def load_page_meta() -> dict[str, tuple[str, str]]:
    """Year/block of every page in the raw corpus (source_ids refer to raw page ids)."""
    return {d.id: (d.metadata.get("year", ""), d.metadata.get("block", "")) for d in load_documents(config.CORPUS_PATH)}


def load_items(golden: Path, page_meta: dict[str, tuple[str, str]]) -> list[dict]:
    items = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    for item in items:
        item["segment"] = segment(item["source_ids"], page_meta)
    return items


def print_coverage(goldens: list[Path], page_meta: dict[str, tuple[str, str]]) -> None:
    items = [item for g in goldens for item in load_items(g, page_meta)]
    table = coverage(items, page_meta)
    segments = site_segments(page_meta)
    print("Questions per question type x segment, dev/test (- = none)\n")
    print("| type | " + " | ".join(segments) + " |")
    print("|---" * (len(segments) + 1) + "|")
    for qtype, row in table.items():
        cells = [f"{c['dev']}/{c['test']}" if c["dev"] or c["test"] else "-" for c in row.values()]
        print(f"| {qtype} | " + " | ".join(cells) + " |")


def print_run(run: str, golden: Path, page_meta: dict[str, tuple[str, str]]) -> None:
    """Per-group numbers of a saved dev run. Labels come from the question file, so older runs can be grouped too."""
    labels = {item["id"]: item for item in load_items(golden, page_meta)}
    results = [json.loads(line) for line in (EVALS_DIR / "results" / f"{run}.jsonl").read_text(encoding="utf-8").splitlines()]
    if any(labels[r["id"]].get("split") != "dev" for r in results):
        raise SystemExit("per-group numbers are for the dev split only; report test as totals")
    for r in results:
        r["question_type"], r["segment"] = labels[r["id"]]["question_type"], labels[r["id"]]["segment"]
    print(json.dumps({"by_question_type": group_summary(results, "question_type"),
                      "by_segment": group_summary(results, "segment")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="name of a saved dev run in evals/results/")
    parser.add_argument("--golden", help="question file (default: both sets for coverage, page-style for --run)")
    args = parser.parse_args()
    page_meta = load_page_meta()
    if args.run:
        print_run(args.run, Path(args.golden or EVALS_DIR / "golden_set.jsonl"), page_meta)
    else:
        goldens = [Path(args.golden)] if args.golden else [EVALS_DIR / "golden_set.jsonl", EVALS_DIR / "golden_student.jsonl"]
        print_coverage(goldens, page_meta)


if __name__ == "__main__":
    main()
