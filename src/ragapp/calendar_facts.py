"""Calendar facts for the answering model: the dates a question refers to, looked up in the academic calendar by code.

    calendar_facts("What are the dates of week 2?", student) ->
        ["Today, Tue 24 Nov 2026, is in Year 2 Block B, teaching week 3 (Mon 23 Nov 2026 - Fri 27 Nov 2026).",
         "Year 2 Block B (the student's block): teaching week 1 starts Mon 9 Nov 2026; ...",
         "Week 2 of Year 2 Block B: Mon 16 Nov 2026 - Fri 20 Nov 2026."]

Always: today (its teaching week, or the break it is in with the teaching weeks around it) and the student's own
block. Then one fact per time or block reference in the question. A reference is any run of words the resolver
understands (week N, next / last week, a date, tomorrow, a weekday, next / last block or semester, a named block),
found by trying every run of up to MAX_WORDS words, longest first, so the forms are the resolver's own and are not
repeated here. Two forms are skipped in running text because they usually mean something else there: weekday
abbreviations ("I sat the exam") and day-month with a hyphen ("tasks 1-2").

The facts go into the prompt as trusted context (generate.py), like the Student line. Nothing here calls an LLM.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

from ragapp.resolver import (WEEKDAYS, BlockTarget, DateRange, day_info, load_calendar, resolve_block,
                             resolve_time, teaching_weeks, week_number)
from ragapp.student import StudentContext

MAX_WORDS = 5  # the longest form: "year 1 block c" (4), "the 2nd of december" (4)
WORD = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")
FULL_DAY_NAMES = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
AMBIGUOUS = re.compile(r"\d{1,2}-\d{1,2}|" + "|".join(w for w in WEEKDAYS if w not in FULL_DAY_NAMES))


@dataclass(frozen=True)
class Reference:
    kind: str    # "week" (week N), "time" (a day or calendar week), "block"
    phrase: str  # as the resolver reads it: "week 2", "next block", "8 february"


def find_references(question: str, ctx: StudentContext) -> list[Reference]:
    """The time and block references in a question, in order: the longest run of words the resolver accepts."""
    words = WORD.findall(question.lower())
    found, i = [], 0
    while i < len(words):
        for n in range(min(MAX_WORDS, len(words) - i), 0, -1):
            phrase = re.sub(r"^(?:on|the) ", "", " ".join(words[i:i + n]))
            kind = _kind(phrase, ctx)
            if kind:
                found.append(Reference(kind, phrase))
                i += n
                break
        else:
            i += 1
    return found


def _kind(phrase: str, ctx: StudentContext) -> str | None:
    if AMBIGUOUS.fullmatch(phrase):
        return None
    if week_number(phrase) is not None:
        return "week"
    if resolve_time(phrase, ctx.today) is not None:
        return "time"
    if resolve_block(phrase, ctx.today, ctx.year, ctx.block) is not None:
        return "block"
    return None


def calendar_facts(question: str, ctx: StudentContext) -> list[str]:
    """Today, the student's block, then a fact per reference in the question; each fact once."""
    own = resolve_block("", ctx.today, ctx.year, ctx.block) if ctx.year else None
    refs = find_references(question, ctx)
    named = [resolve_block(r.phrase, ctx.today, ctx.year, ctx.block) for r in refs if r.kind == "block"]
    week_target = next((t for t in named if t and t.blocks), own)  # "week 5 of block c": weeks of the block named

    facts = [_today(ctx)]
    if own:
        facts += _block(own, ctx, "", own=True)
    for ref in refs:
        if ref.kind == "block":
            target = resolve_block(ref.phrase, ctx.today, ctx.year, ctx.block)
            if target and target.blocks and target != own:
                relative = re.match(r"(next|last|previous) ", ref.phrase)
                facts += _block(target, ctx, ref.phrase.capitalize() if relative else "")
        elif ref.kind == "week" and week_target:
            facts.append(_week_n(week_number(ref.phrase), week_target, ctx))
        elif ref.kind == "time":
            span = resolve_time(ref.phrase, ctx.today)
            label = "" if re.search(r"\d", ref.phrase) else ref.phrase.capitalize()
            facts.append(_span(span, ctx.year, label))
    return list(dict.fromkeys(facts))


# ---------------------------------------------------------------- writing the facts


def _d(day: date) -> str:
    return f"{day:%a} {day.day} {day:%b %Y}"


def _range(start: date, end: date) -> str:
    return f"{_d(start)} - {_d(end)}"


def _name(year: str, segment: str) -> str:
    """('Year2', 'BlockB') -> 'Year 2 Block B'; without a year just 'Block B'."""
    readable = re.sub(r"^(Block|Semester)(\w)$", r"\1 \2", segment)
    return f"{year.replace('Year', 'Year ')} {readable}" if year else readable


def _period(day: date):
    return next((p for y in load_calendar() for p in y.periods if p.start <= day <= p.end), None)


def _closed_days(start: date, end: date) -> list[str]:
    return [f"{_d(d)} ({name})" for y in load_calendar() for d, name in sorted(y.closed_days.items())
            if start <= d <= end]


def _teaching_week_from(day: date, step: int, year: str):
    """The first day of a teaching week found walking from `day` (excluded) in steps of one day; None at the
    edge of the calendar file."""
    for _ in range(366):
        day += timedelta(days=step)
        info = day_info(day, year)
        if info.kind == "outside":
            return None
        if info.week is not None:
            return info
    return None


def _today(ctx: StudentContext) -> str:
    info = day_info(ctx.today, ctx.year)
    head = f"Today, {_d(ctx.today)},"
    if info.kind == "outside":
        return f"{head} is not in the academic calendar the assistant has."
    where = _name(ctx.year, info.block)
    if info.week is None:
        p = _period(ctx.today)
        text = f"{head} is in the {p.name} ({_range(p.start, p.end)}): no teaching, so no week number."
        before = _teaching_week_from(p.start, -1, ctx.year)
        after = _teaching_week_from(p.end, 1, ctx.year)
        if before:
            monday = before.day - timedelta(days=before.day.weekday())
            text += (f" Last teaching week before it: {_name(ctx.year, before.block)} week {before.week}"
                     f" ({_range(monday, monday + timedelta(days=4))}).")
        if after:
            text += f" Next teaching week: {_name(ctx.year, after.block)} week {after.week}, from {_d(after.day)}."
        if info.closed_day:
            text += f" BUas is closed today ({info.closed_day})."
        return text
    monday = ctx.today - timedelta(days=ctx.today.weekday())
    week = f"{where}, teaching week {info.week} ({_range(monday, monday + timedelta(days=4))})"
    if info.kind == "weekend":
        return f"{head} is a weekend day in {week}."
    text = f"{head} is in {week}." + (f" Week note: {info.note}." if info.note else "")
    return text + (f" BUas is closed today ({info.closed_day})." if info.kind == "closed_day" else "")


def _block(target: BlockTarget, ctx: StudentContext, label: str, own: bool = False) -> list[str]:
    """One fact per segment: first and last teaching week, end, breaks and closed days inside it."""
    facts = []
    for segment in target.blocks:
        name = _name(target.year, segment)
        head = f"{label} = {name}" if label else name
        unit = "semester" if segment.startswith("Semester") else "block"
        head += f" (the student's {unit})" if own else ""
        weeks = teaching_weeks(BlockTarget(target.year, (segment,), target.years_ahead), ctx.today)
        if not weeks:
            facts.append(f"{head}: its dates are not in the calendar yet.")
            continue
        year = next(y for y in load_calendar() if y.start <= weeks[0].monday <= y.end)
        letters = [b for b in year.blocks.values() if any(w in b.weeks for w in weeks)]
        start, end = letters[0].start, letters[-1].end
        last = weeks[-1]
        text = (f"{head}: teaching week 1 starts {_d(weeks[0].monday)}; last teaching week, week {len(weeks)}: "
                f"{_range(last.monday, last.monday + timedelta(days=4))}" + (f" ({last.note})" if last.note else "")
                + f"; the {unit} ends {_d(end)}")
        text += f" (the regulations give {_d(letters[-1].ter_end)})." if letters[-1].ter_end else "."
        breaks = [f"{p.name} ({_range(p.start, p.end)})" for p in year.periods if start <= p.start <= end]
        if breaks:
            text += " No teaching: " + ", ".join(breaks) + "."
        if closed := _closed_days(start, end):
            text += " BUas closed: " + ", ".join(closed) + "."
        facts.append(text)
    return facts


def _week_n(n: int, target: BlockTarget, ctx: StudentContext) -> str:
    names = " + ".join(_name(target.year, s) for s in target.blocks)
    weeks = teaching_weeks(target, ctx.today)
    if not weeks:
        return f"Week {n} of {names}: its dates are not in the calendar yet."
    if not 1 <= n <= len(weeks):
        return f"{names} has teaching weeks 1-{len(weeks)}; there is no week {n}."
    week = weeks[n - 1]
    friday = week.monday + timedelta(days=4)
    text = f"Week {n} of {names}: {_range(week.monday, friday)}" + (f" ({week.note})." if week.note else ".")
    if closed := _closed_days(week.monday, friday):
        text += " BUas closed: " + ", ".join(closed) + "."
    return text


def _span(span: DateRange, year: str, label: str) -> str:
    """A single day, or a Monday-Friday calendar week ("next week")."""
    head = f"{label} = " if label else ""
    info = day_info(span.start, year)
    if span.start == span.end:
        head += _d(span.start)
        if info.kind == "outside":
            return f"{head}: not in the academic calendar the assistant has."
        if info.week is None:
            p = _period(span.start)
            text = f"{head}: {p.name}, no teaching ({_range(p.start, p.end)})."
            return text + (f" BUas closed ({info.closed_day})." if info.closed_day else "")
        where = f"{_name(year, info.block)} teaching week {info.week}"
        if info.kind == "weekend":
            return f"{head}: weekend, in {where}."
        if info.kind == "closed_day":
            return f"{head}: BUas closed ({info.closed_day}), in {where}."
        return f"{head}: {where}." if not info.note else f"{head}: {where} ({info.note})."
    head += _range(span.start, span.end)
    if info.kind == "outside":
        return f"{head}: not in the academic calendar the assistant has."
    if info.week is None:
        p = _period(span.start)
        return f"{head}: {p.name}, no teaching ({_range(p.start, p.end)})."
    text = f"{head}: {_name(year, info.block)} teaching week {info.week}" + (f" ({info.note})." if info.note else ".")
    if closed := _closed_days(span.start, span.end):
        text += " BUas closed: " + ", ".join(closed) + "."
    return text
