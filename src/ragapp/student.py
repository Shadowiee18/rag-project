"""Who is asking, and when: the context that turns "what do i have to do today" into something searchable."""

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from ragapp import config
from ragapp.documents import load_documents

# A timeline row on a block page, e.g. "W2 16.11–20.11 | Topic Alpha | Tasks 1-2" (week, dates, then the plan)
WEEK_ROW = re.compile(r"^W(\d{1,2}) (\d{2})\.(\d{2})[–-](\d{2})\.(\d{2})\s*\|(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class StudentContext:
    today: date
    year: str = ""   # site folder format, e.g. "Year2"
    block: str = ""  # e.g. "BlockB"


def nearest_year(day: int, month: int, today: date) -> date:
    """Timeline dates have no year and blocks cross New Year: pick the year that puts the date closest to today."""
    options = [date(today.year + offset, month, day) for offset in (-1, 0, 1)]
    return min(options, key=lambda d: abs((d - today).days))


def find_week(page_text: str, today: date) -> dict | None:
    """Return {'week': 2, 'start': date, 'end': date, 'plan': '...'} for the timeline row containing today."""
    for m in WEEK_ROW.finditer(page_text):
        start = nearest_year(int(m.group(2)), int(m.group(3)), today)
        end = nearest_year(int(m.group(4)), int(m.group(5)), today)
        if start <= today <= end:
            plan = " | ".join(part.strip() for part in m.group(6).split("|") if part.strip())
            return {"week": int(m.group(1)), "start": start, "end": end, "plan": plan}
    return None


@lru_cache(maxsize=1)
def _pages() -> dict[str, str]:
    return {doc.id: doc.text for doc in load_documents(config.CORPUS_PATH)}


def search_phrase(ctx: StudentContext) -> str:
    """Words for a search query, e.g. 'Year 2 Block B week 2'. Added by code, never by the LLM."""
    if not ctx.year:
        return ""
    parts = [ctx.year.replace("Year", "Year "), re.sub(r"^(Block|Semester)(\w)$", r"\1 \2", ctx.block) if ctx.block else ""]
    week = find_week(_pages().get(f"{ctx.year}/{ctx.block}", ""), ctx.today) if ctx.block else None
    if week:
        parts.append(f"week {week['week']}")
    return " ".join(p for p in parts if p)


def describe(ctx: StudentContext, with_plan: bool = True) -> str:
    """One or two plain sentences the LLM can use, e.g.
    'Today is Wednesday 18 November 2026. The student is in Year 2, Block B, week 2 (16.11–20.11): Topic Alpha | Tasks 1-2.'

    with_plan=False leaves out the week's plan (": Topic Alpha | Tasks 1-2"). The query rewriter gets that
    version, because a small model copies those words into unrelated search queries.
    """
    text = f"Today is {ctx.today.strftime('%A')} {ctx.today.day} {ctx.today.strftime('%B %Y')}."
    if not ctx.year:
        return text
    readable_year = ctx.year.replace("Year", "Year ")
    readable_block = re.sub(r"^(Block|Semester)(\w)$", r"\1 \2", ctx.block) if ctx.block else ""
    where = f"{readable_year}, {readable_block}" if readable_block else readable_year
    week = find_week(_pages().get(f"{ctx.year}/{ctx.block}", ""), ctx.today) if ctx.block else None
    if week:
        span = f"{week['start']:%d.%m}–{week['end']:%d.%m}"
        plan = f": {week['plan']}" if with_plan else ""
        return f"{text} The student is in {where}, week {week['week']} ({span}){plan}."
    return f"{text} The student is in {where}."
