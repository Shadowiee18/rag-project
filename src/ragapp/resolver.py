"""Turn time and block words into dates and blocks, in code (no LLM).

    resolve_time("tomorrow" / "wednesday" / "next week" / "week 3" / "14.12", today, target) -> DateRange
    resolve_block("my block" / "next block" / "Year 1 Block C" / "semester 2", today, year, block) -> BlockTarget
    day_info(date, year) -> which block and week a date is in, or which break / closed day it is

The dates come from calendar.yaml (the official academic calendar), never from arithmetic like
"first Monday + 7 * (N - 1)": every block has its break in a different week.
Years and blocks use the site's folder names: "Year2", "BlockB", and "Semester1" for Year 3.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import yaml

CALENDAR_PATH = Path(__file__).with_name("calendar.yaml")

# The site's segments in study order. "next block" after Year 1 Block D is Year 2 Block A, a year later.
SEGMENTS = [("Year1", "BlockA"), ("Year1", "BlockB"), ("Year1", "BlockC"), ("Year1", "BlockD"),
            ("Year2", "BlockA"), ("Year2", "BlockB"), ("Year2", "BlockC"), ("Year2", "BlockD"),
            ("Year3", "Semester1"), ("Year3", "Semester2")]

WEEKDAYS = {"monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1, "wednesday": 2, "wed": 2,
            "thursday": 3, "thu": 3, "thur": 3, "thurs": 3, "friday": 4, "fri": 4,
            "saturday": 5, "sat": 5, "sunday": 6, "sun": 6}
MONTHS = {name: i for i, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}
MONTHS |= {name[:3]: i for name, i in MONTHS.items()} | {"sept": 9}


# ---------------------------------------------------------------- the calendar file


@dataclass(frozen=True)
class Week:
    number: int
    monday: date
    note: str = ""  # e.g. "assessment"


@dataclass(frozen=True)
class Block:
    letter: str  # "A"
    start: date
    end: date
    weeks: tuple[Week, ...]
    ter_end: date | None = None  # the regulations' end date, when it differs from the week-by-week calendar


@dataclass(frozen=True)
class Period:
    """A week or more without teaching: a holiday, the study week, the introduction week, summer."""
    name: str
    kind: str
    start: date
    end: date


@dataclass(frozen=True)
class AcademicYear:
    name: str
    start: date
    end: date
    blocks: dict[str, Block]
    semesters: dict[int, tuple[str, ...]]  # {1: ("A", "B"), 2: ("C", "D")}
    periods: tuple[Period, ...]
    closed_days: dict[date, str]


@lru_cache(maxsize=4)
def load_calendar(path: Path = CALENDAR_PATH) -> tuple[AcademicYear, ...]:
    """All academic years in the file, oldest first. Raises ValueError when the file contradicts itself."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return tuple(sorted((_read_year(y) for y in raw["academic_years"]), key=lambda y: y.start))


def _read_year(raw: dict) -> AcademicYear:
    blocks = {
        letter: Block(letter, b["start"], b["end"],
                      tuple(Week(w["week"], w["monday"], str(w.get("note", ""))) for w in b["weeks"]),
                      b.get("ter_end"))
        for letter, b in raw["blocks"].items()
    }
    year = AcademicYear(
        name=str(raw["name"]), start=raw["start"], end=raw["end"], blocks=blocks,
        semesters={int(k): tuple(v) for k, v in raw["semesters"].items()},
        periods=tuple(Period(p["name"], p["kind"], p["start"], p["end"]) for p in raw["periods"]),
        closed_days=dict(raw["closed_days"] or {}),
    )
    _check(year)
    return year


def _check(year: AcademicYear) -> None:
    for p in year.periods:
        if p.start.weekday() != 0 or p.end.weekday() != 6:
            raise ValueError(f"{year.name} {p.name}: a period runs from a Monday to a Sunday")
    mondays = []
    for b in year.blocks.values():
        for i, w in enumerate(b.weeks, start=1):
            where = f"{year.name} Block {b.letter} week {w.number}"
            friday = w.monday + timedelta(days=4)
            if w.monday.weekday() != 0:
                raise ValueError(f"{where}: {w.monday} is not a Monday")
            if w.number != i:
                raise ValueError(f"{where}: weeks must be numbered 1, 2, 3, ... in order")
            if not b.start <= w.monday <= friday <= b.end:
                raise ValueError(f"{where}: outside the block's start and end")
            for p in year.periods:
                if w.monday <= p.end and p.start <= friday:
                    raise ValueError(f"{where} and {p.name} overlap")
            mondays.append(w.monday)
    if mondays != sorted(set(mondays)):
        raise ValueError(f"{year.name}: teaching weeks must be in date order, without repeats")


def _year_of(day: date) -> AcademicYear | None:
    return next((y for y in load_calendar() if y.start <= day <= y.end), None)


def _letters(year: AcademicYear, segment: str) -> tuple[str, ...]:
    """'BlockB' -> ('B',); 'Semester1' -> ('A', 'B')."""
    return year.semesters[int(segment[-1])] if segment.startswith("Semester") else (segment[-1],)


def _teaching_weeks(year: AcademicYear, segments: tuple[str, ...]) -> list[Week]:
    """The teaching weeks of one or more segments, in order. Week N of a semester is the Nth in this list."""
    return [w for s in segments for letter in _letters(year, s) for w in year.blocks[letter].weeks]


def _segment(letter: str, study_year: str, year: AcademicYear) -> str:
    """Block letter -> the folder for this study year: 'BlockC', or 'Semester2' for Year 3."""
    if study_year == "Year3":
        return next(f"Semester{n}" for n, letters in year.semesters.items() if letter in letters)
    return f"Block{letter}"


# ---------------------------------------------------------------- which block / week is a date in


@dataclass(frozen=True)
class DayInfo:
    day: date
    kind: str  # teaching | weekend | closed_day | holiday | study_week | intro_week | closure_week |
    #            finalisation_week | summer_holiday | no_teaching | outside (not in calendar.yaml)
    block: str = ""            # "BlockB", or "Semester1" for Year 3; "" outside the blocks
    week: int | None = None    # teaching week within the block (Year 3: within the semester)
    name: str = ""             # name of the break or closed day
    closed_day: str = ""       # set when BUas is closed that day, also inside a holiday
    note: str = ""             # e.g. "assessment"


def day_info(day: date, study_year: str = "") -> DayInfo:
    """What kind of day this is in the academic calendar. study_year="Year3" counts in semesters."""
    year = _year_of(day)
    if year is None:
        return DayInfo(day, "outside")
    closed = year.closed_days.get(day, "")
    block = next((b for b in year.blocks.values() if b.start <= day <= b.end), None)
    segment = _segment(block.letter, study_year, year) if block else ""

    period = next((p for p in year.periods if p.start <= day <= p.end), None)
    if period:
        return DayInfo(day, period.kind, segment, None, period.name, closed)

    week = next((w for w in block.weeks if w.monday <= day <= w.monday + timedelta(days=6)), None) if block else None
    if week is None:  # a gap in the file; the tests make sure there is none
        return DayInfo(day, "outside")
    number = [w.monday for w in _teaching_weeks(year, (segment,))].index(week.monday) + 1
    kind = "weekend" if day.weekday() >= 5 else "closed_day" if closed else "teaching"
    return DayInfo(day, kind, segment, number, closed, closed, week.note)


# ---------------------------------------------------------------- time references -> date range


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date  # inclusive; a single day has start == end


@dataclass(frozen=True)
class BlockTarget:
    """The year and block(s) a question is about, in folder names (usable as a search filter).

    blocks is empty for a whole year ("Year 1"); two blocks for "semester 1" in a block year.
    years_ahead: 0 = this academic year's dates, 1 = next ("next block" after Block D), -1 = last.
    """
    year: str
    blocks: tuple[str, ...]
    years_ahead: int = 0


def resolve_time(ref: str, today: date, target: BlockTarget | None = None) -> DateRange | None:
    """A time reference -> the dates it means. None when the words are not one of the forms below.

    today, tomorrow; a weekday (the next one, today included); this / next / last week (Monday-Friday of
    that calendar week); week N (of the target block, from calendar.yaml); a date (14.12, 14 december,
    2026-12-14; a date without a year is taken in the academic year of today).
    """
    ref = _clean(ref)
    if ref == "today":
        return DateRange(today, today)
    if ref == "tomorrow":
        return _one_day(today + timedelta(days=1))
    if ref in WEEKDAYS:
        return _one_day(today + timedelta(days=(WEEKDAYS[ref] - today.weekday()) % 7))
    if m := re.fullmatch(r"(this|next|last) week", ref):
        offset = {"this": 0, "next": 1, "last": -1}[m.group(1)]
        monday = today - timedelta(days=today.weekday()) + timedelta(weeks=offset)
        return DateRange(monday, monday + timedelta(days=4))
    if (n := week_number(ref)) is not None:
        return _week_n(n, today, target)
    day = _parse_date(ref, today)
    return _one_day(day) if day else None


def _one_day(day: date) -> DateRange:
    return DateRange(day, day)


def _clean(ref: str) -> str:
    """Lower case, single spaces, without a leading 'on' or 'the'."""
    return re.sub(r"^(?:on|the) ", "", " ".join(ref.lower().split()))


def week_number(ref: str) -> int | None:
    """'week 3' / 'wk 3' / 'w3' -> 3; None for anything else."""
    m = re.fullmatch(r"(?:week|wk|w) ?(\d{1,2})", _clean(ref))
    return int(m.group(1)) if m else None


def teaching_weeks(target: BlockTarget, today: date) -> list[Week] | None:
    """The target's teaching weeks in order (week N is the Nth; a Year 3 semester runs on through both blocks),
    from the academic year it is in. None when that year is not in calendar.yaml, or for a whole year."""
    if not target.blocks:
        return None
    years = load_calendar()
    now = next((i for i, y in enumerate(years) if y.start <= today <= y.end), None)
    if now is None or not 0 <= now + target.years_ahead < len(years):
        return None
    return _teaching_weeks(years[now + target.years_ahead], target.blocks)


def _week_n(n: int, today: date, target: BlockTarget | None) -> DateRange | None:
    weeks = teaching_weeks(target, today) if target else None
    if not weeks or not 1 <= n <= len(weeks):
        return None
    monday = weeks[n - 1].monday
    return DateRange(monday, monday + timedelta(days=4))


def _parse_date(ref: str, today: date) -> date | None:
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", ref):
        y, mo, d = (int(g) for g in m.groups())
    elif m := re.fullmatch(r"(\d{1,2})[./-](\d{1,2})(?:[./-](\d{4}))?", ref):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    elif (m := re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)? (?:of )?([a-z]+)(?: (\d{4}))?", ref)) and m.group(2) in MONTHS:
        d, mo, y = int(m.group(1)), MONTHS[m.group(2)], int(m.group(3) or 0)
    elif (m := re.fullmatch(r"([a-z]+) (\d{1,2})(?:st|nd|rd|th)?(?:,? (\d{4}))?", ref)) and m.group(1) in MONTHS:
        d, mo, y = int(m.group(2)), MONTHS[m.group(1)], int(m.group(3) or 0)
    else:
        return None
    if y:
        try:
            return date(y, mo, d)
        except ValueError:  # 31.04, 29.02 in a non-leap year
            return None
    return _guess_year(d, mo, today)


def _guess_year(d: int, mo: int, today: date) -> date | None:
    """A day and month without a year: the one inside today's academic year (nearest to today if two are).
    Outside the calendar file: the nearest to today. Impossible dates (29.02 in a non-leap year) give None."""
    candidates = []
    for y in (today.year - 1, today.year, today.year + 1):
        try:
            candidates.append(date(y, mo, d))
        except ValueError:
            pass
    year = _year_of(today)
    if year:
        candidates = [c for c in candidates if year.start <= c <= year.end]
    return min(candidates, key=lambda c: abs((c - today).days), default=None)


# ---------------------------------------------------------------- block references -> year/block


def resolve_block(ref: str, today: date, year: str, block: str = "") -> BlockTarget | None:
    """A block reference -> the year/block it means, from the student's year and block.

    "" / my / this / current block -> the student's own; next / last / previous block -> the neighbour in
    SEGMENTS (across the academic year after Block D); next / last / previous semester -> the semester next to the
    student's (Year 1-2: its two blocks); named: "Year 1 Block C", "block c" (own year),
    "Y1C", "semester 2", "Year 1". Without a block in the context, the block is taken from today's date.
    None when the reference cannot be resolved.
    """
    ref = _clean(ref)
    own = _own_segment(today, year, block)
    if ref in ("", "my block", "this block", "current block"):
        return BlockTarget(year, (own,)) if own else None
    if m := re.fullmatch(r"(next|last|previous) block", ref):
        if not own:
            return None
        i = SEGMENTS.index((year, own)) + (1 if m.group(1) == "next" else -1)
        if not 0 <= i < len(SEGMENTS):
            return None
        target_year, target_block = SEGMENTS[i]
        return BlockTarget(target_year, (target_block,), int(target_year[-1]) - int(year[-1]))
    if m := re.fullmatch(r"(next|last|previous) semester", ref):
        return _next_semester(own, year, today, 1 if m.group(1) == "next" else -1) if own else None
    return _named_block(ref, year, today)


def _next_semester(own: str, year: str, today: date, step: int) -> BlockTarget | None:
    """The semester before or after the student's own, across study years (Year 1 semester 2 -> Year 2 semester 1)."""
    calendar = _year_of(today) or load_calendar()[-1]
    semester = int(own[-1]) if own.startswith("Semester") else next(
        n for n, letters in calendar.semesters.items() if own[-1] in letters)
    number, semester = divmod((int(year[-1]) - 1) * 2 + (semester - 1) + step, 2)  # count semesters from Year 1
    if not 0 <= number <= 2:
        return None
    target_year = f"Year{number + 1}"
    blocks = ((f"Semester{semester + 1}",) if target_year == "Year3"
              else tuple(f"Block{letter}" for letter in calendar.semesters[semester + 1]))
    return BlockTarget(target_year, blocks, number + 1 - int(year[-1]))


def block_ref_in_question(question: str) -> str:
    """'what do we have next block?' -> 'next block'; '' when the question has no relative block."""
    m = re.search(r"\b(next|last|previous) block\b", question.lower())
    return m.group(0) if m else ""


def segments_for(year: str, blocks: list[str]) -> list[str]:
    """Block names -> the folders that year uses: Year 3 'BlockC' -> 'Semester2';
    Year 1-2 'Semester1' -> 'BlockA', 'BlockB'. Other names stay as they are."""
    calendar = load_calendar()[-1]  # the block -> semester split is the same every year
    out: list[str] = []
    for b in blocks:
        if year == "Year3" and b.startswith("Block"):
            names = [_segment(b[-1], year, calendar)]
        elif year in {"Year1", "Year2"} and b.startswith("Semester"):
            names = [f"Block{letter}" for letter in calendar.semesters[int(b[-1])]]
        else:
            names = [b]
        out += [n for n in names if n not in out]
    return out


def _own_segment(today: date, year: str, block: str) -> str:
    """The student's segment as a folder name, or "" if unknown. Year 3 given as 'BlockC' -> 'Semester2'."""
    if not block:
        return day_info(today, year).block if year in {"Year1", "Year2", "Year3"} else ""
    if year == "Year3" and block.startswith("Block"):
        block = _segment(block[-1], year, _year_of(today) or load_calendar()[-1])
    return block if (year, block) in SEGMENTS else ""


def _named_block(ref: str, year: str, today: date) -> BlockTarget | None:
    if m := re.fullmatch(r"y([123])([a-d])", ref):
        number, letter, semester = m.group(1), m.group(2), None
    elif m := re.fullmatch(r"(?:year ?([123]) )?block ([a-d])", ref):
        number, letter, semester = m.group(1), m.group(2), None
    elif m := re.fullmatch(r"(?:year ?([123]) )?semester ([12])", ref):
        number, letter, semester = m.group(1), None, int(m.group(2))
    elif m := re.fullmatch(r"year ?([123])", ref):
        return BlockTarget(f"Year{m.group(1)}", ())
    else:
        return None

    target_year = f"Year{number}" if number else year
    if not target_year:
        return None  # "block c" without a year: every year has one
    calendar = _year_of(today) or load_calendar()[-1]
    if letter:
        return BlockTarget(target_year, (_segment(letter.upper(), target_year, calendar),))
    if target_year == "Year3":
        return BlockTarget(target_year, (f"Semester{semester}",))
    return BlockTarget(target_year, tuple(f"Block{x}" for x in calendar.semesters[semester]))
