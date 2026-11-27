"""A small held-out check for calendar questions: wordings that are NOT the behavioural suite's templates.

The suite has no test split, and the calendar facts (CALENDAR_FACTS=1) were built while looking at its families, so
these 13 questions check that the gain is not tied to the suite's sentences. Written before the first run and not
changed after. Two have no time words at all (the answer must come from today's facts). Gold dates are typed by hand
from the 2026-27 academic calendar. Scored with the suite's own code (behavioral.score_answer / outcome); the answers
are also read by Claude, because "the first week / date the answer names" is strict for free-form questions.

Run (with the main checkout's data, see evals/README.md):
    CALENDAR_FACTS=0 uv run python evals/calendar_sanity.py --name sanity_off
    CALENDAR_FACTS=1 uv run python evals/calendar_sanity.py --name sanity_on
"""

import argparse
import json
from datetime import date

from behavioral import RESULTS_DIR, outcome, right_wrong_abstained, run

D = date.fromisoformat


def _case(i, today, year, block, question, gold):
    family = "week" if ("week" in gold or "break" in gold) else "day"
    return {"id": f"sanity-{i:02d}", "segment": f"{year} {block}", "year": year, "block": block, "today": today,
            "question": question, "family": family, "gold": gold}


CASES = [
    _case(1, "2026-11-24", "Year2", "BlockB", "When is week 5 of block C?", {"dates": ["2027-03-08", "2027-03-12"]}),
    _case(2, "2026-12-15", "Year2", "BlockB", "Is next week a holiday?", {"break": "Christmas holiday"}),
    _case(3, "2027-03-02", "Year2", "BlockC", "What's the first day of block D?", {"date": "2027-04-12"}),
    _case(4, "2026-11-24", "Year3", "Semester1", "When does semester 2 begin?", {"date": "2027-02-01"}),
    _case(5, "2026-11-24", "Year2", "BlockB", "What week will it be on 14 January?", {"week": 8}),
    _case(6, "2027-01-27", "Year1", "BlockB", "Are we in a teaching week or on holiday right now?",
          {"break": "study week"}),
    _case(7, "2026-11-10", "Year1", "BlockB", "Which teaching week is the 2nd of December in?", {"week": 4}),
    _case(8, "2026-12-01", "Year2", "BlockB", "What's the date range for week 7?",
          {"dates": ["2027-01-04", "2027-01-08"]}),
    _case(9, "2027-02-17", "Year2", "BlockC", "When did the previous block's first week start?", {"date": "2026-11-09"}),
    _case(10, "2026-12-23", "Year2", "BlockB", "When do classes start again after the holiday?", {"date": "2027-01-04"}),
    _case(11, "2027-02-17", "Year3", "Semester2", "Which semester week is 10 March in?", {"week": 5}),
    _case(12, "2026-10-14", "Year1", "BlockA", "Do we have lessons on 21 October?", {"break": "autumn holiday"}),
    _case(13, "2026-09-15", "Year1", "BlockA", "On what date do Block B lectures begin?", {"date": "2026-11-09"}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    from ragapp import config
    from ragapp.rag import answer as rag_answer
    from ragapp.student import StudentContext

    def pipeline(case: dict) -> dict:
        out = rag_answer(case["question"], config.TOP_K, StudentContext(D(case["today"]), case["year"], case["block"]))
        return {"answer": out["answer"], "chunks": out["chunks"], "calendar": out.get("calendar", [])}

    RESULTS_DIR.mkdir(exist_ok=True)
    results = [r | {"outcome": outcome(r)} for r in run(CASES, pipeline)]
    with open(RESULTS_DIR / f"{args.name}_sanity.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"cases": len(results), "right_wrong_abstained": right_wrong_abstained(results),
               "calendar_facts": config.CALENDAR_FACTS, "llm_model": config.LLM_MODEL}
    (RESULTS_DIR / f"{args.name}_sanity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
