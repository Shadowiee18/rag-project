"""Behavioural suite (Phase 3 step 3): does the assistant handle time and blocks in EVERY year and block?

The normal eval asks all its student questions as Year 2 Block B on two dates in November, so a feature that
works for one block looks fine. This suite asks the same template questions for all 10 segments (Year 1-2
Blocks A-D, Year 3 Semesters 1-2) on several dates each (week 1, week 3, the last teaching week, every break),
in the CheckList style (Ribeiro et al., ACL 2020):

  MFT  minimum functionality: "which week are we in?", "what are the dates of week N?",
       "when does the next block start?"                       -> exact match on the week / dates
  DIR  directional: "what do we have next block?" / "what did we do last block?"
       -> the top search result must come from that other block's pages
  INV  invariance: two wordings of the week question must give the same week

Gold answers are computed HERE from calendar.yaml, with this file's own small code, not with the app's resolver,
so a bug in the resolver cannot make its own test pass. The questions are templates (no course text), and the gold
needs no hand labels. Scoring is code only (no LLM judge).

Run:  uv run python evals/behavioral.py --name step4         (uses Ollama + Qdrant: one answer per case)
      uv run python evals/behavioral.py --list               (print the cases, no model needed)
Writes evals/results/<name>_behavioral.jsonl (answers, git-ignored) and <name>_behavioral_summary.json.
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import yaml

EVALS_DIR = Path(__file__).resolve().parent
CALENDAR_PATH = EVALS_DIR.parent / "src" / "ragapp" / "calendar.yaml"
RESULTS_DIR = EVALS_DIR / "results"

SEGMENTS = [("Year1", "BlockA"), ("Year1", "BlockB"), ("Year1", "BlockC"), ("Year1", "BlockD"),
            ("Year2", "BlockA"), ("Year2", "BlockB"), ("Year2", "BlockC"), ("Year2", "BlockD"),
            ("Year3", "Semester1"), ("Year3", "Semester2")]
FAMILIES = ("week", "week_dates", "next_start", "dir_next", "dir_last")

D = date.fromisoformat


# ---------------------------------------------------------------- gold from the calendar (independent code)


@lru_cache(maxsize=1)
def _year() -> dict:
    """The 2026-27 entry of calendar.yaml: the suite is written for this academic year."""
    raw = yaml.safe_load(CALENDAR_PATH.read_text(encoding="utf-8"))
    return next(y for y in raw["academic_years"] if str(y["name"]) == "2026-27")


def _letters(block: str) -> list[str]:
    return list(_year()["semesters"][int(block[-1])]) if block.startswith("Semester") else [block[-1]]


def teaching_mondays(year: str, block: str) -> list[date]:
    """Monday of every teaching week of the segment, in order (a Year 3 semester runs on through both blocks)."""
    blocks = _year()["blocks"]
    return [w["monday"] for letter in _letters(block) for w in blocks[letter]["weeks"]]


def segment_span(year: str, block: str) -> tuple[date, date]:
    blocks = _year()["blocks"]
    letters = _letters(block)
    return blocks[letters[0]]["start"], blocks[letters[-1]]["end"]


def _breaks(year: str, block: str) -> list[dict]:
    start, end = segment_span(year, block)
    return [p for p in _year()["periods"] if start <= p["start"] <= end]


def _week_gold(year: str, block: str, day: date) -> dict:
    for p in _breaks(year, block):
        if p["start"] <= day <= p["end"]:
            return {"break": p["name"]}
    mondays = teaching_mondays(year, block)
    return {"week": next(i for i, m in enumerate(mondays, start=1) if m <= day <= m + timedelta(days=6))}


def make_cases() -> list[dict]:
    cases = []
    for i, (year, block) in enumerate(SEGMENTS):
        seg = f"{year} {block}"
        unit = "semester" if block.startswith("Semester") else "block"
        mondays = teaching_mondays(year, block)
        base = {"segment": seg, "year": year, "block": block}

        # MFT: which week (two wordings each, for INV), on week 1, week 3, the last week and every break
        days = [mondays[0] + timedelta(days=1), mondays[2] + timedelta(days=1), mondays[-1] + timedelta(days=3)]
        days += [p["start"] + timedelta(days=2) for p in _breaks(year, block)]
        for day in days:
            group = f"week-{year}-{block}-{day}"
            for suffix, question in (("a", f"Which week of the {unit} are we in?"), ("b", "What week is it right now?")):
                cases.append({**base, "id": f"{group}-{suffix}", "family": "week", "today": str(day),
                              "question": question, "gold": _week_gold(year, block, day), "inv_group": group})

        today = mondays[2] + timedelta(days=1)  # Tuesday of week 3: an ordinary teaching day
        for n in (2, len(mondays)):
            monday = mondays[n - 1]
            cases.append({**base, "id": f"dates-{year}-{block}-w{n}", "family": "week_dates", "today": str(today),
                          "question": f"What are the dates of week {n}?",
                          "gold": {"dates": [str(monday), str(monday + timedelta(days=4))]}})

        after = SEGMENTS[i + 1] if i + 1 < len(SEGMENTS) else None
        before = SEGMENTS[i - 1] if i > 0 else None
        if after and after[0] == year:  # the next block of the same study year is in this year's calendar
            cases.append({**base, "id": f"next-{year}-{block}", "family": "next_start", "today": str(today),
                          "question": f"When does the next {unit} start?",
                          "gold": {"date": str(teaching_mondays(*after)[0])}})
        if after:
            cases.append({**base, "id": f"dirnext-{year}-{block}", "family": "dir_next", "today": str(today),
                          "question": "What do we have next block?", "gold": {"target": list(after)}})
        if before:
            cases.append({**base, "id": f"dirlast-{year}-{block}", "family": "dir_last", "today": str(today),
                          "question": "What did we do last block?", "gold": {"target": list(before)}})
    return cases


# ---------------------------------------------------------------- reading answers (exact match)

WEEK = re.compile(r"\b(?:week\s*|w)(\d{1,2})\b", re.IGNORECASE)
BREAK_WORDS = re.compile(
    r"\b(holiday|break|vacation|recess|study week|no (?:teaching|classes|lectures))\b", re.IGNORECASE
)
_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                "november", "december"]
MONTHS = {m: i for i, m in enumerate(_MONTH_NAMES, start=1)} | {m[:3]: i for i, m in enumerate(_MONTH_NAMES, 1)}
MONTHS["sept"] = 9
_M = "(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b"
_ORD = r"(?:st|nd|rd|th)?"
DATE = re.compile(
    r"\b(?P<iso>(\d{4})-(\d{1,2})-(\d{1,2}))\b"
    rf"|\b(?P<range>(\d{{1,2}}){_ORD}\s*[–-]\s*(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_M})"
    rf"|\b(?P<dm>(\d{{1,2}}){_ORD}\s+(?:of\s+)?{_M})"
    rf"|\b(?P<md>{_M}\s+(\d{{1,2}}){_ORD})\b"
    r"|\b(?P<num>(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?)\b",
    re.IGNORECASE,
)


def extract_weeks(text: str) -> list[int]:
    """Week numbers in order of appearance: 'week 3', 'Week 12', 'W5'."""
    return [int(n) for n in WEEK.findall(text)]


def extract_dates(text: str) -> list[tuple[int, int]]:
    """(day, month) of every date in the text, in order: 23.11, 23/11/2026, 23 November, November 23rd,
    2026-11-23, and both ends of '23-27 November'. Years are ignored (the answer may leave them out)."""
    found = []
    for m in DATE.finditer(text):
        g = [x.lower() if isinstance(x, str) else x for x in m.groups()]
        if m.group("iso"):
            pairs = [(int(g[3]), int(g[2]))]
        elif m.group("range"):
            pairs = [(int(g[5]), MONTHS[g[7]]), (int(g[6]), MONTHS[g[7]])]
        elif m.group("dm"):
            pairs = [(int(g[9]), MONTHS[g[10]])]
        elif m.group("md"):
            pairs = [(int(g[13]), MONTHS[g[12]])]
        else:
            pairs = [(int(g[15]), int(g[16]))]
        found += [(d, mo) for d, mo in pairs if 1 <= d <= 31 and 1 <= mo <= 12]
    return found


def _day_month(iso: str) -> tuple[int, int]:
    return D(iso).day, D(iso).month


def score_answer(gold: dict, answer: str) -> bool:
    """week: the first week the answer names is the right one. break: the answer says there is no teaching.
    dates: the first two dates are that week's Monday and Friday. date: the first date is the right one."""
    if "week" in gold:
        weeks = extract_weeks(answer)
        return bool(weeks) and weeks[0] == gold["week"]
    if "break" in gold:
        return bool(BREAK_WORDS.search(answer))
    dates = extract_dates(answer)
    if "dates" in gold:
        return len(dates) >= 2 and set(dates[:2]) == {_day_month(d) for d in gold["dates"]}
    return bool(dates) and dates[0] == _day_month(gold["date"])


def _as_list(value) -> list[str]:
    return value if isinstance(value, list) else [value]


def score_retrieval(target: list[str], chunks: list[dict]) -> bool:
    """DIR: the top search result is a page of the target year and block (a merged copy of several blocks counts)."""
    if not chunks:
        return False
    top = chunks[0]
    return target[0] in _as_list(top.get("year", "")) and target[1] in _as_list(top.get("block", ""))


def outcome(result: dict) -> str | None:
    """right / wrong / abstained for a case with an answer to read (None for the DIR search checks).
    A failed case is "wrong" when the answer names a week (week family) or a date (date families): it committed to
    an answer. Otherwise it abstained ("I don't know", "the context has no calendar"). Wrong costs a student more
    than abstaining, so the CRAG score counts +1 / -1 / 0."""
    if "target" in result["gold"]:
        return None
    if result["passed"]:
        return "right"
    named = extract_weeks(result["answer"]) if result["family"] == "week" else extract_dates(result["answer"])
    return "wrong" if named else "abstained"


def right_wrong_abstained(results: list[dict]) -> dict:
    """Per family (the week family split into teaching days and breaks), and the total with its CRAG score."""
    rows: dict[str, dict] = defaultdict(lambda: {"right": 0, "wrong": 0, "abstained": 0})
    for r in results:
        if (kind := outcome(r)) is None:
            continue
        family = r["family"]
        if family == "week":
            family = "week (breaks)" if "break" in r["gold"] else "week (teaching days)"
        rows[family][kind] += 1
        rows["total"][kind] += 1
    total = rows["total"]
    n = sum(total.values())
    total["crag"] = round((total["right"] - total["wrong"]) / n, 2) if n else 0.0
    return dict(rows)


def week_reading(answer: str) -> str | None:
    """What a week answer says, for the INV check: 'week 3', 'break' or None."""
    weeks = extract_weeks(answer)
    if weeks:
        return f"week {weeks[0]}"
    return "break" if BREAK_WORDS.search(answer) else None


# ---------------------------------------------------------------- running and summarising


def run(cases: list[dict], answer_fn) -> list[dict]:
    """answer_fn(case) -> {"answer": str, "chunks": [chunk dicts, best first]}."""
    results = []
    for case in cases:
        out = answer_fn(case)
        chunks = out["chunks"]
        if "target" in case["gold"]:
            passed = score_retrieval(case["gold"]["target"], chunks)
        else:
            passed = score_answer(case["gold"], out["answer"])
        results.append({**case, "answer": out["answer"], "passed": passed, "calendar": out.get("calendar", []),
                        "reading": week_reading(out["answer"]) if case["family"] == "week" else None,
                        "top_pages": [c.get("doc_id") for c in chunks[:5]]})
    return results


def _share(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 3) if flags else 0.0


def summarise(results: list[dict]) -> dict:
    by_family, by_segment, by_cell = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in results:
        by_family[r["family"]].append(r["passed"])
        by_segment[r["segment"]].append(r["passed"])
        by_cell[f"{r['family']} | {r['segment']}"].append(r["passed"])
    groups = defaultdict(list)
    for r in results:
        if r.get("inv_group"):
            groups[r["inv_group"]].append(r["reading"])
    pairs = [readings for readings in groups.values() if len(readings) > 1]
    return {
        "cases": len(results),
        "passed": _share([r["passed"] for r in results]),
        "per_family": {f: _share(v) for f, v in by_family.items()},
        "per_segment": {s: _share(v) for s, v in by_segment.items()},
        "per_family_segment": {c: f"{sum(v)}/{len(v)}" for c, v in by_cell.items()},
        "right_wrong_abstained": right_wrong_abstained(results),
        "invariance": {
            "pairs": len(pairs),
            # both wordings give the same reading, and it is a reading at all (not two non-answers)
            "consistent": _share([len(set(p)) == 1 and p[0] is not None for p in pairs]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="run name, e.g. step4")
    parser.add_argument("--limit", type=int, help="only the first N cases (a quick check)")
    parser.add_argument("--list", action="store_true", help="print the cases and stop (no model needed)")
    args = parser.parse_args()
    cases = make_cases()[: args.limit] if args.limit else make_cases()
    if args.list:
        for c in cases:
            print(c["id"], "|", c["today"], "|", c["question"], "|", c["gold"])
        print(len(cases), "cases")
        return
    if not args.name:
        parser.error("--name is needed for a run")

    from ragapp import config
    from ragapp.rag import answer as rag_answer
    from ragapp.student import StudentContext, describe

    def pipeline(case: dict) -> dict:
        student = StudentContext(D(case["today"]), case["year"], case["block"])
        out = rag_answer(case["question"], config.TOP_K, student)
        return {"answer": out["answer"], "chunks": out["chunks"], "student": describe(student),
                "calendar": out.get("calendar", [])}

    RESULTS_DIR.mkdir(exist_ok=True)
    results = run(cases, pipeline)
    with open(RESULTS_DIR / f"{args.name}_behavioral.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(results) | {"config": {
        "query_rewrite": config.QUERY_REWRITE, "hybrid": config.HYBRID, "rerank_kind": config.RERANK_KIND,
        "rerank_candidates": config.RERANK_CANDIDATES, "collection": config.COLLECTION, "llm_backend": config.LLM_BACKEND, "student_line_as_source": config.STUDENT_LINE_AS_SOURCE, "llm_model": config.LLM_MODEL,
        "scope_mode": config.SCOPE_MODE, "scope_boost_block": config.SCOPE_BOOST_BLOCK, "scope_boost_year": config.SCOPE_BOOST_YEAR,
        "calendar_facts": config.CALENDAR_FACTS,
    }}
    (RESULTS_DIR / f"{args.name}_behavioral_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("cases", "passed", "per_family", "right_wrong_abstained", "invariance")},
                     indent=2))


if __name__ == "__main__":
    main()
