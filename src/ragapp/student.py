"""Who is asking, and when: the context that turns "what do i have to do today" into something searchable."""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

from ragapp import config
from ragapp.documents import load_documents
from ragapp.resolver import DayInfo, day_info, segments_for

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


def _calendar_day(ctx: StudentContext) -> DayInfo | None:
    """Today in the academic calendar, if the calendar puts today in the student's block (else None)."""
    if not ctx.year or not ctx.block:
        return None
    info = day_info(ctx.today, ctx.year)
    return info if info.block in segments_for(ctx.year, [ctx.block]) else None


def readable_name(year: str, blocks: list[str]) -> str:
    """Folder names -> words: ('Year2', ['BlockC']) -> 'Year 2 Block C'; two blocks -> 'Year 1 Block A and Block B'."""
    names = [re.sub(r"^(Block|Semester)(\w)$", r"\1 \2", b) for b in blocks]
    return f"{year.replace('Year', 'Year ')} {' and '.join(names)}".strip()


def search_phrase(ctx: StudentContext) -> str:
    """Words for a search query, e.g. 'Year 2 Block B week 2'. Added by code, never by the LLM.
    The week comes from the academic calendar; in a break or study week there is no week."""
    if not ctx.year:
        return ""
    parts = [ctx.year.replace("Year", "Year "), re.sub(r"^(Block|Semester)(\w)$", r"\1 \2", ctx.block) if ctx.block else ""]
    info = _calendar_day(ctx)
    if info and info.week:
        parts.append(f"week {info.week}")
    return " ".join(p for p in parts if p)


def describe(ctx: StudentContext, with_plan: bool = True) -> str:
    """One or two plain sentences the LLM can use, e.g.
    'Today is Wednesday 18 November 2026. The student is in Year 2, Block B, week 2 (16.11–20.11): Topic Alpha | Tasks 1-2.'

    Week number and dates come from the academic calendar (every block); a break is named instead
    ("This week is the Christmas holiday."). The week's plan comes from the block page's timeline row,
    only where the page has one for the same week.
    with_plan=False leaves out the week's plan (": Topic Alpha | Tasks 1-2"). The query rewriter gets that
    version, because a small model copies those words into unrelated search queries.
    """
    text = f"Today is {ctx.today.strftime('%A')} {ctx.today.day} {ctx.today.strftime('%B %Y')}."
    if not ctx.year:
        return text
    readable_year = ctx.year.replace("Year", "Year ")
    readable_block = re.sub(r"^(Block|Semester)(\w)$", r"\1 \2", ctx.block) if ctx.block else ""
    where = f"{readable_year}, {readable_block}" if readable_block else readable_year
    info = _calendar_day(ctx)
    if info is None:
        return f"{text} The student is in {where}."
    if info.week is None:
        return f"{text} The student is in {where}." + (f" This week is the {info.name}." if info.name else "")
    monday = ctx.today - timedelta(days=ctx.today.weekday())
    span = f"{monday:%d.%m}–{monday + timedelta(days=4):%d.%m}"
    plan = ""
    if with_plan:
        row = find_week(_pages().get(f"{ctx.year}/{ctx.block}", ""), ctx.today)
        if row and row["week"] == info.week:
            plan = f": {row['plan']}"
    closed = f" BUas is closed today ({info.closed_day})." if info.kind == "closed_day" else ""
    return f"{text} The student is in {where}, week {info.week} ({span}){plan}.{closed}"
