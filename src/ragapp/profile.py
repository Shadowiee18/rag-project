"""The student profile: the study year is asked once per academic year; block and week come from today's date.

A profile stores the year ("Year2") and the academic year the answer was given in ("2026-27").
In a new academic year the student is asked again (they may have moved up a year).
"""

import re
from datetime import date

from ragapp.resolver import day_info, load_calendar, segments_for
from ragapp.student import StudentContext, no_block_reason

YEARS = ("Year1", "Year2", "Year3")


def academic_year(today: date) -> str:
    """'2026-27'. From calendar.yaml; outside the file, the Dutch academic year (1 September - 31 August),
    so a new year is still noticed when the file has not been updated yet."""
    known = next((y for y in load_calendar() if y.start <= today <= y.end), None)
    if known:
        return known.name
    first = today.year if today.month >= 9 else today.year - 1
    return f"{first}-{(first + 1) % 100:02d}"


def needs_year(year: str, given_in: str, today: date) -> bool:
    """Ask the study year when there is none, or it was given in another academic year."""
    return year not in YEARS or given_in != academic_year(today)


def student_context(year: str, today: date, block: str = "") -> StudentContext:
    """The student's context for a question. block "" = the block the calendar puts today in
    ("" in the summer or outside calendar.yaml: unknown, not guessed). A given block (testing, evals) wins."""
    return StudentContext(today, year, block or day_info(today, year).block)


def status(year: str, given_in: str, today: date, block: str = "") -> dict:
    """What the chat page shows: whether to ask the year, and where the student is today.
    label: e.g. 'Year 2 · Block B · week 3', 'Year 2 · block unknown (summer holiday)'."""
    out = {"ask_year": needs_year(year, given_in, today), "academic_year": academic_year(today),
           "year": year or None, "block": None, "week": None, "label": ""}
    if out["ask_year"]:
        return out
    ctx = student_context(year, today, block)
    info = day_info(today, year)
    same_block = ctx.block and info.block in segments_for(year, [ctx.block])
    parts = [_readable(year)]
    if not ctx.block:
        parts.append(f"block unknown ({no_block_reason(today, year)})")
    else:
        parts.append(_readable(ctx.block))
        if same_block and info.week:
            parts.append(f"week {info.week}")
        elif same_block and info.name:
            parts.append(info.name)
    out |= {"block": ctx.block or None, "week": info.week if same_block else None, "label": " · ".join(parts)}
    return out


def _readable(name: str) -> str:
    """'Year2' -> 'Year 2', 'BlockB' -> 'Block B', 'Semester1' -> 'Semester 1'."""
    return re.sub(r"^(Year|Block|Semester)(\w)$", r"\1 \2", name)
