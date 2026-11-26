"""Calendar file + resolver: turn "tomorrow", "week 3", "next block" into dates and blocks, in code.

The expected dates below are typed in by hand from the 2026-27 academic calendar (TER Article 3.3 and the
programme's year calendar), independently of calendar.yaml, so these tests check the file as well as the code.
No Ollama, Qdrant or course data needed.
"""

from datetime import date, timedelta

import pytest

from ragapp.resolver import BlockTarget, DateRange, day_info, load_calendar, resolve_block, resolve_time

D = date.fromisoformat
TODAY = D("2026-11-26")  # a Thursday in Year 2 Block B week 3

# Monday of every teaching week, per block (week 1 first). Breaks sit in a different place in every block.
TEACHING_WEEKS = {
    "A": ["2026-08-31", "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28", "2026-10-05", "2026-10-12",
          "2026-10-26", "2026-11-02"],
    "B": ["2026-11-09", "2026-11-16", "2026-11-23", "2026-11-30", "2026-12-07", "2026-12-14", "2027-01-04",
          "2027-01-11", "2027-01-18"],
    "C": ["2027-02-01", "2027-02-15", "2027-02-22", "2027-03-01", "2027-03-08", "2027-03-15", "2027-03-22",
          "2027-03-29", "2027-04-05"],
    "D": ["2027-04-12", "2027-04-19", "2027-05-03", "2027-05-10", "2027-05-17", "2027-05-24", "2027-05-31",
          "2027-06-07", "2027-06-14", "2027-06-21"],
}


def _year():
    return load_calendar()[0]


# ---------------------------------------------------------------- the calendar file itself


@pytest.mark.parametrize(
    "block,number,monday",
    [(b, i + 1, m) for b, mondays in TEACHING_WEEKS.items() for i, m in enumerate(mondays)],
)
def test_teaching_week_mondays_match_the_official_calendar(block, number, monday):
    week = _year().blocks[block].weeks[number - 1]
    assert (week.number, week.monday) == (number, D(monday))


@pytest.mark.parametrize("block,count", [("A", 9), ("B", 9), ("C", 9), ("D", 10)])
def test_number_of_teaching_weeks_per_block(block, count):
    assert len(_year().blocks[block].weeks) == count


@pytest.mark.parametrize(
    "block,start,end",
    [("A", "2026-08-31", "2026-11-08"), ("B", "2026-11-09", "2027-01-31"),
     ("C", "2027-02-01", "2027-04-11"), ("D", "2027-04-12", "2027-06-27")],
)
def test_block_spans(block, start, end):
    b = _year().blocks[block]
    assert (b.start, b.end) == (D(start), D(end))


def test_block_d_keeps_the_ter_end_date_next_to_the_calendar_week_10():
    assert _year().blocks["D"].ter_end == D("2027-06-18")


@pytest.mark.parametrize(
    "block,number,note",
    [("A", 9, "assessment"), ("B", 9, "assessment"), ("C", 9, "assessment"), ("D", 8, "assessment"),
     ("D", 9, "resit"), ("D", 10, "resit"), ("B", 3, "")],
)
def test_week_notes(block, number, note):
    assert note in _year().blocks[block].weeks[number - 1].note
    if not note:
        assert _year().blocks[block].weeks[number - 1].note == ""


@pytest.mark.parametrize(
    "kind,start,end",
    [
        ("intro_week", "2026-08-24", "2026-08-30"),
        ("holiday", "2026-10-19", "2026-10-25"),         # autumn
        ("holiday", "2026-12-21", "2027-01-03"),         # Christmas, two weeks
        ("study_week", "2027-01-25", "2027-01-31"),
        ("holiday", "2027-02-08", "2027-02-14"),         # spring
        ("holiday", "2027-04-26", "2027-05-02"),         # May
        ("closure_week", "2027-06-28", "2027-07-04"),
        ("finalisation_week", "2027-07-05", "2027-07-11"),
        ("summer_holiday", "2027-07-12", "2027-08-15"),
        ("no_teaching", "2027-08-16", "2027-08-22"),     # staff back, students not yet
        ("intro_week", "2027-08-23", "2027-08-29"),      # next year's Year 1
    ],
)
def test_periods_without_teaching(kind, start, end):
    assert any((p.kind, p.start, p.end) == (kind, D(start), D(end)) for p in _year().periods)


@pytest.mark.parametrize(
    "day,weekday",
    [("2027-01-01", 4), ("2027-02-08", 0), ("2027-03-26", 4), ("2027-03-29", 0), ("2027-04-26", 0),
     ("2027-04-27", 1), ("2027-05-05", 2), ("2027-05-06", 3), ("2027-05-07", 4), ("2027-05-17", 0)],
)
def test_closed_days(day, weekday):
    closed = _year().closed_days
    assert D(day) in closed and closed[D(day)]
    assert D(day).weekday() == weekday


def test_exactly_ten_closed_days():
    assert len(_year().closed_days) == 10


def test_every_week_starts_on_monday_and_weeks_increase():
    mondays = [w.monday for b in "ABCD" for w in _year().blocks[b].weeks]
    assert all(m.weekday() == 0 for m in mondays)
    assert mondays == sorted(set(mondays))


def test_every_day_of_the_year_is_a_teaching_week_or_a_named_period():
    year = _year()
    day = year.start
    while day <= year.end:
        assert day_info(day).kind != "outside", day
        day += timedelta(days=1)


def test_loader_rejects_a_week_that_does_not_start_on_monday(tmp_path):
    bad = tmp_path / "cal.yaml"
    bad.write_text(_mini_calendar(week1="2026-09-01"))  # a Tuesday
    with pytest.raises(ValueError, match="Monday"):
        load_calendar(bad)


def test_loader_rejects_a_teaching_week_inside_a_break(tmp_path):
    bad = tmp_path / "cal.yaml"
    bad.write_text(_mini_calendar(week1="2026-08-31", holiday="2026-08-31"))
    with pytest.raises(ValueError, match="overlap"):
        load_calendar(bad)


def test_loader_accepts_a_small_valid_file(tmp_path):
    good = tmp_path / "cal.yaml"
    good.write_text(_mini_calendar(week1="2026-08-31"))
    assert load_calendar(good)[0].blocks["A"].weeks[0].monday == D("2026-08-31")


def _mini_calendar(week1: str, holiday: str = "2026-09-07") -> str:
    end = (D(holiday) + timedelta(days=6)).isoformat()
    return f"""
academic_years:
  - name: test
    start: 2026-08-31
    end: 2026-09-13
    semesters: {{1: [A], 2: []}}
    blocks:
      A:
        start: 2026-08-31
        end: 2026-09-13
        weeks:
          - {{week: 1, monday: {week1}}}
    periods:
      - {{name: test holiday, kind: holiday, start: {holiday}, end: {end}}}
    closed_days: {{}}
"""


# ---------------------------------------------------------------- which block / week is a date in


@pytest.mark.parametrize(
    "day,kind,block,week",
    [
        ("2026-08-23", "outside", "", None),        # before the calendar
        ("2026-08-24", "intro_week", "", None),
        ("2026-08-30", "intro_week", "", None),
        ("2026-08-31", "teaching", "BlockA", 1),
        ("2026-09-04", "teaching", "BlockA", 1),
        ("2026-09-05", "weekend", "BlockA", 1),
        ("2026-10-16", "teaching", "BlockA", 7),
        ("2026-10-19", "holiday", "BlockA", None),  # autumn holiday
        ("2026-10-25", "holiday", "BlockA", None),
        ("2026-10-26", "teaching", "BlockA", 8),
        ("2026-11-02", "teaching", "BlockA", 9),
        ("2026-11-08", "weekend", "BlockA", 9),
        ("2026-11-09", "teaching", "BlockB", 1),
        ("2026-11-26", "teaching", "BlockB", 3),
        ("2026-12-18", "teaching", "BlockB", 6),
        ("2026-12-21", "holiday", "BlockB", None),  # Christmas
        ("2026-12-25", "holiday", "BlockB", None),
        ("2026-12-31", "holiday", "BlockB", None),
        ("2027-01-01", "holiday", "BlockB", None),  # New Year, also a closed day
        ("2027-01-03", "holiday", "BlockB", None),
        ("2027-01-04", "teaching", "BlockB", 7),
        ("2027-01-22", "teaching", "BlockB", 9),
        ("2027-01-25", "study_week", "BlockB", None),
        ("2027-01-31", "study_week", "BlockB", None),
        ("2027-02-01", "teaching", "BlockC", 1),
        ("2027-02-08", "holiday", "BlockC", None),  # spring holiday, also Carnival
        ("2027-02-15", "teaching", "BlockC", 2),
        ("2027-03-26", "closed_day", "BlockC", 7),  # Good Friday
        ("2027-03-29", "closed_day", "BlockC", 8),  # Easter Monday
        ("2027-04-09", "teaching", "BlockC", 9),
        ("2027-04-11", "weekend", "BlockC", 9),
        ("2027-04-12", "teaching", "BlockD", 1),
        ("2027-04-26", "holiday", "BlockD", None),  # May holiday, also King's Day
        ("2027-05-03", "teaching", "BlockD", 3),
        ("2027-05-05", "closed_day", "BlockD", 3),
        ("2027-05-06", "closed_day", "BlockD", 3),
        ("2027-05-07", "closed_day", "BlockD", 3),
        ("2027-05-17", "closed_day", "BlockD", 5),
        ("2027-06-18", "teaching", "BlockD", 9),
        ("2027-06-21", "teaching", "BlockD", 10),
        ("2027-06-27", "weekend", "BlockD", 10),
        ("2027-06-28", "closure_week", "", None),
        ("2027-07-05", "finalisation_week", "", None),
        ("2027-07-12", "summer_holiday", "", None),
        ("2027-08-13", "summer_holiday", "", None),
        ("2027-08-16", "no_teaching", "", None),
        ("2027-08-23", "intro_week", "", None),
        ("2027-08-30", "outside", "", None),        # 2027-28 is not in the file yet
        ("2028-02-29", "outside", "", None),        # leap day, no crash
        ("2024-02-29", "outside", "", None),
    ],
)
def test_day_info(day, kind, block, week):
    info = day_info(D(day))
    assert (info.kind, info.block, info.week) == (kind, block, week)


@pytest.mark.parametrize(
    "day,year,kind,block,week",
    [
        ("2026-08-31", "Year3", "teaching", "Semester1", 1),
        ("2026-11-02", "Year3", "teaching", "Semester1", 9),
        ("2026-11-09", "Year3", "teaching", "Semester1", 10),
        ("2026-11-26", "Year3", "teaching", "Semester1", 12),
        ("2026-12-28", "Year3", "holiday", "Semester1", None),
        ("2027-01-22", "Year3", "teaching", "Semester1", 18),
        ("2027-01-25", "Year3", "study_week", "Semester1", None),
        ("2027-02-01", "Year3", "teaching", "Semester2", 1),
        ("2027-04-12", "Year3", "teaching", "Semester2", 10),
        ("2027-06-21", "Year3", "teaching", "Semester2", 19),
        ("2027-07-01", "Year3", "closure_week", "", None),
        ("2027-03-01", "Year1", "teaching", "BlockC", 4),
        ("2027-03-01", "Year2", "teaching", "BlockC", 4),
    ],
)
def test_day_info_per_year(day, year, kind, block, week):
    info = day_info(D(day), year)
    assert (info.kind, info.block, info.week) == (kind, block, week)


@pytest.mark.parametrize(
    "day,word",
    [("2026-10-20", "autumn"), ("2026-12-28", "christmas"), ("2027-01-27", "study"), ("2027-02-10", "spring"),
     ("2027-04-28", "may"), ("2027-05-05", "liberation"), ("2027-03-26", "good friday"), ("2026-08-25", "year 1")],
)
def test_day_info_names_the_break_or_closed_day(day, word):
    assert word in day_info(D(day)).name.lower()


@pytest.mark.parametrize(
    "day,closed",
    [("2027-01-01", "new year"), ("2027-02-08", "carnival"), ("2027-04-27", "king"), ("2026-12-25", "")],
)
def test_closed_day_inside_a_holiday_is_still_reported(day, closed):
    assert closed in day_info(D(day)).closed_day.lower()
    if not closed:
        assert day_info(D(day)).closed_day == ""


def test_assessment_week_note():
    assert "assessment" in day_info(D("2026-11-04")).note


# ---------------------------------------------------------------- time references -> date range

Y2B = BlockTarget("Year2", ("BlockB",))


def _r(start, end=None):
    return DateRange(D(start), D(end or start))


@pytest.mark.parametrize(
    "today,ref,expected",
    [
        ("2026-11-26", "today", _r("2026-11-26")),
        ("2026-11-26", "  Today ", _r("2026-11-26")),
        ("2026-11-26", "tomorrow", _r("2026-11-27")),
        ("2026-11-27", "tomorrow", _r("2026-11-28")),       # Friday -> Saturday, just the date
        ("2026-12-31", "tomorrow", _r("2027-01-01")),       # New Year
        ("2027-02-28", "tomorrow", _r("2027-03-01")),
        ("2028-02-28", "tomorrow", _r("2028-02-29")),       # leap year
        ("2028-02-29", "tomorrow", _r("2028-03-01")),
        ("2028-02-29", "today", _r("2028-02-29")),
        ("2026-11-26", "wednesday", _r("2026-12-02")),      # the next Wednesday
        ("2026-11-26", "thursday", _r("2026-11-26")),       # today is Thursday
        ("2026-11-26", "friday", _r("2026-11-27")),
        ("2026-11-26", "monday", _r("2026-11-30")),
        ("2026-11-26", "on Monday", _r("2026-11-30")),
        ("2026-11-26", "mon", _r("2026-11-30")),
        ("2026-11-26", "wed", _r("2026-12-02")),
        ("2026-11-26", "thurs", _r("2026-11-26")),
        ("2026-11-28", "monday", _r("2026-11-30")),         # from a Saturday
        ("2026-12-31", "friday", _r("2027-01-01")),
        ("2028-02-29", "wednesday", _r("2028-03-01")),
        ("2026-11-26", "this week", _r("2026-11-23", "2026-11-27")),
        ("2026-11-23", "this week", _r("2026-11-23", "2026-11-27")),
        ("2026-11-29", "this week", _r("2026-11-23", "2026-11-27")),  # Sunday: the calendar week it is in
        ("2026-11-26", "next week", _r("2026-11-30", "2026-12-04")),
        ("2026-11-26", "last week", _r("2026-11-16", "2026-11-20")),
        ("2026-12-17", "next week", _r("2026-12-21", "2026-12-25")),  # into the Christmas break
        ("2026-12-30", "next week", _r("2027-01-04", "2027-01-08")),  # across New Year
        ("2027-01-05", "last week", _r("2026-12-28", "2027-01-01")),
        ("2028-02-29", "this week", _r("2028-02-28", "2028-03-03")),
    ],
)
def test_relative_days_and_weeks(today, ref, expected):
    assert resolve_time(ref, D(today), Y2B) == expected


@pytest.mark.parametrize(
    "year,blocks,week,start,end",
    [
        ("Year1", ("BlockA",), 1, "2026-08-31", "2026-09-04"),
        ("Year1", ("BlockA",), 8, "2026-10-26", "2026-10-30"),  # after the autumn holiday
        ("Year1", ("BlockB",), 7, "2027-01-04", "2027-01-08"),  # after Christmas
        ("Year1", ("BlockC",), 2, "2027-02-15", "2027-02-19"),  # after the spring holiday
        ("Year1", ("BlockD",), 3, "2027-05-03", "2027-05-07"),  # after the May holiday
        ("Year2", ("BlockA",), 9, "2026-11-02", "2026-11-06"),
        ("Year2", ("BlockB",), 1, "2026-11-09", "2026-11-13"),
        ("Year2", ("BlockB",), 3, "2026-11-23", "2026-11-27"),
        ("Year2", ("BlockC",), 9, "2027-04-05", "2027-04-09"),
        ("Year2", ("BlockD",), 10, "2027-06-21", "2027-06-25"),
        ("Year3", ("Semester1",), 1, "2026-08-31", "2026-09-04"),
        ("Year3", ("Semester1",), 10, "2026-11-09", "2026-11-13"),
        ("Year3", ("Semester1",), 18, "2027-01-18", "2027-01-22"),
        ("Year3", ("Semester2",), 1, "2027-02-01", "2027-02-05"),
        ("Year3", ("Semester2",), 10, "2027-04-12", "2027-04-16"),
        ("Year3", ("Semester2",), 19, "2027-06-21", "2027-06-25"),
        ("Year2", ("BlockA", "BlockB"), 10, "2026-11-09", "2026-11-13"),  # "semester 1" for a block year
    ],
)
def test_week_n_in_every_segment(year, blocks, week, start, end):
    assert resolve_time(f"week {week}", TODAY, BlockTarget(year, blocks)) == _r(start, end)


@pytest.mark.parametrize("ref", ["week 3", "Week 3", "wk 3", "w3", "W3", "week3"])
def test_week_n_spellings(ref):
    assert resolve_time(ref, TODAY, Y2B) == _r("2026-11-23", "2026-11-27")


@pytest.mark.parametrize(
    "ref,target",
    [
        ("week 10", BlockTarget("Year2", ("BlockB",))),
        ("week 11", BlockTarget("Year2", ("BlockD",))),
        ("week 19", BlockTarget("Year3", ("Semester1",))),
        ("week 20", BlockTarget("Year3", ("Semester2",))),
        ("week 0", BlockTarget("Year2", ("BlockB",))),
        ("week 3", None),                                    # no block to count in
        ("week 3", BlockTarget("Year1", ())),                # a whole year: ambiguous
        ("week 3", BlockTarget("Year2", ("BlockA",), 1)),    # next academic year: not in the file
        ("week 3", BlockTarget("Year1", ("BlockD",), -1)),   # previous academic year: not in the file
    ],
)
def test_week_n_that_does_not_exist_gives_none(ref, target):
    assert resolve_time(ref, TODAY, target) is None


def test_week_n_when_today_is_outside_the_calendar():
    assert resolve_time("week 3", D("2025-11-26"), Y2B) is None
    assert resolve_time("week 3", D("2028-02-29"), Y2B) is None


@pytest.mark.parametrize(
    "today,ref,expected",
    [
        ("2026-11-26", "14.12", "2026-12-14"),
        ("2026-11-26", "14-12", "2026-12-14"),
        ("2026-11-26", "14/12", "2026-12-14"),
        ("2026-11-26", "14 december", "2026-12-14"),
        ("2026-11-26", "December 14", "2026-12-14"),
        ("2026-11-26", "14th december", "2026-12-14"),
        ("2026-11-26", "14 dec", "2026-12-14"),
        ("2026-11-26", "4.1", "2027-01-04"),                 # January is in this academic year
        ("2026-11-26", "04.01.2027", "2027-01-04"),
        ("2026-11-26", "2027-01-04", "2027-01-04"),
        ("2026-11-26", "15 june", "2027-06-15"),
        ("2026-11-26", "1 september", "2026-09-01"),
        ("2026-11-26", "25 august", "2026-08-25"),           # both Augusts are in the file: nearest wins
        ("2027-08-20", "25 august", "2027-08-25"),
        ("2028-02-29", "29.02", "2028-02-29"),               # today outside the file: nearest valid date
        ("2027-12-01", "29 february", "2028-02-29"),
        ("2025-03-01", "1 march", "2025-03-01"),
    ],
)
def test_dates(today, ref, expected):
    assert resolve_time(ref, D(today), Y2B) == _r(expected)


@pytest.mark.parametrize(
    "today,ref",
    [
        ("2026-11-26", "29.02"),        # 2026-27 has no 29 February (the old nearest_year crash)
        ("2026-11-26", "29 february"),
        ("2026-11-26", "2027-02-29"),
        ("2026-11-26", "31.04"),
        ("2026-11-26", "32.12"),
        ("2026-11-26", "14.13"),
        ("2026-11-26", ""),
        ("2026-11-26", "soon"),
        ("2026-11-26", "next month"),
        ("2026-11-26", "next wednesday"),  # ambiguous: not resolved by rules
    ],
)
def test_unresolvable_time_references_give_none(today, ref):
    assert resolve_time(ref, D(today), Y2B) is None


# ---------------------------------------------------------------- block references -> year/block


SEGMENTS = [("Year1", "BlockA"), ("Year1", "BlockB"), ("Year1", "BlockC"), ("Year1", "BlockD"),
            ("Year2", "BlockA"), ("Year2", "BlockB"), ("Year2", "BlockC"), ("Year2", "BlockD"),
            ("Year3", "Semester1"), ("Year3", "Semester2")]


@pytest.mark.parametrize("year,block", SEGMENTS)
def test_my_block_is_the_students_own(year, block):
    assert resolve_block("my block", TODAY, year, block) == BlockTarget(year, (block,), 0)


@pytest.mark.parametrize(
    "year,block,expected",
    [
        ("Year1", "BlockA", BlockTarget("Year1", ("BlockB",), 0)),
        ("Year1", "BlockB", BlockTarget("Year1", ("BlockC",), 0)),
        ("Year1", "BlockC", BlockTarget("Year1", ("BlockD",), 0)),
        ("Year1", "BlockD", BlockTarget("Year2", ("BlockA",), 1)),      # across the academic year
        ("Year2", "BlockA", BlockTarget("Year2", ("BlockB",), 0)),
        ("Year2", "BlockB", BlockTarget("Year2", ("BlockC",), 0)),
        ("Year2", "BlockC", BlockTarget("Year2", ("BlockD",), 0)),
        ("Year2", "BlockD", BlockTarget("Year3", ("Semester1",), 1)),
        ("Year3", "Semester1", BlockTarget("Year3", ("Semester2",), 0)),
        ("Year3", "Semester2", None),                                    # nothing after Year 3 on the site
    ],
)
def test_next_block(year, block, expected):
    assert resolve_block("next block", TODAY, year, block) == expected


@pytest.mark.parametrize(
    "year,block,expected",
    [
        ("Year1", "BlockA", None),
        ("Year1", "BlockB", BlockTarget("Year1", ("BlockA",), 0)),
        ("Year1", "BlockC", BlockTarget("Year1", ("BlockB",), 0)),
        ("Year1", "BlockD", BlockTarget("Year1", ("BlockC",), 0)),
        ("Year2", "BlockA", BlockTarget("Year1", ("BlockD",), -1)),
        ("Year2", "BlockB", BlockTarget("Year2", ("BlockA",), 0)),
        ("Year2", "BlockC", BlockTarget("Year2", ("BlockB",), 0)),
        ("Year2", "BlockD", BlockTarget("Year2", ("BlockC",), 0)),
        ("Year3", "Semester1", BlockTarget("Year2", ("BlockD",), -1)),
        ("Year3", "Semester2", BlockTarget("Year3", ("Semester1",), 0)),
    ],
)
def test_last_block(year, block, expected):
    assert resolve_block("last block", TODAY, year, block) == expected


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("", BlockTarget("Year2", ("BlockB",))),             # no reference = my own block
        ("this block", BlockTarget("Year2", ("BlockB",))),
        ("current block", BlockTarget("Year2", ("BlockB",))),
        ("The next block", BlockTarget("Year2", ("BlockC",))),
        ("  Next Block ", BlockTarget("Year2", ("BlockC",))),
        ("previous block", BlockTarget("Year2", ("BlockA",))),
        ("the last block", BlockTarget("Year2", ("BlockA",))),
    ],
)
def test_block_reference_spellings(ref, expected):
    assert resolve_block(ref, TODAY, "Year2", "BlockB") == expected


@pytest.mark.parametrize(
    "ref,year,block,expected",
    [
        ("year 1 block c", "Year2", "BlockB", BlockTarget("Year1", ("BlockC",))),
        ("Year 1 Block C", "Year3", "Semester1", BlockTarget("Year1", ("BlockC",))),
        ("y1c", "Year2", "BlockB", BlockTarget("Year1", ("BlockC",))),
        ("block c", "Year2", "BlockB", BlockTarget("Year2", ("BlockC",))),     # the student's own year
        ("block a", "Year3", "Semester2", BlockTarget("Year3", ("Semester1",))),  # Year 3 has semesters
        ("block d", "Year3", "Semester1", BlockTarget("Year3", ("Semester2",))),
        ("semester 2", "Year3", "Semester1", BlockTarget("Year3", ("Semester2",))),
        ("semester 1", "Year2", "BlockC", BlockTarget("Year2", ("BlockA", "BlockB"))),
        ("semester 2", "Year1", "BlockA", BlockTarget("Year1", ("BlockC", "BlockD"))),
        ("year 3 semester 1", "Year1", "BlockB", BlockTarget("Year3", ("Semester1",))),
        ("year 3 block b", "Year2", "BlockB", BlockTarget("Year3", ("Semester1",))),
        ("y3a", "Year2", "BlockB", BlockTarget("Year3", ("Semester1",))),
        ("year 1", "Year2", "BlockB", BlockTarget("Year1", ())),
        ("block e", "Year2", "BlockB", None),
        ("year 4", "Year2", "BlockB", None),
        ("semester 3", "Year3", "Semester1", None),
        ("the block after christmas", "Year2", "BlockB", None),  # not a rule-based reference
    ],
)
def test_named_blocks(ref, year, block, expected):
    assert resolve_block(ref, TODAY, year, block) == expected


def test_year3_context_given_as_a_block_letter_is_read_as_its_semester():
    assert resolve_block("my block", TODAY, "Year3", "BlockC") == BlockTarget("Year3", ("Semester2",))
    assert resolve_block("next block", TODAY, "Year3", "BlockB") == BlockTarget("Year3", ("Semester2",))


@pytest.mark.parametrize(
    "today,year,ref,expected",
    [
        ("2026-11-26", "Year2", "my block", BlockTarget("Year2", ("BlockB",))),
        ("2026-11-26", "Year2", "next block", BlockTarget("Year2", ("BlockC",))),
        ("2026-10-20", "Year1", "my block", BlockTarget("Year1", ("BlockA",))),   # in the autumn holiday
        ("2027-01-27", "Year2", "next block", BlockTarget("Year2", ("BlockC",))),  # in the study week
        ("2026-11-26", "Year3", "my block", BlockTarget("Year3", ("Semester1",))),
        ("2026-11-26", "Year3", "next block", BlockTarget("Year3", ("Semester2",))),
        ("2027-06-21", "Year1", "next block", BlockTarget("Year2", ("BlockA",), 1)),
        ("2027-07-15", "Year2", "my block", None),           # summer: no block to take from the date
        ("2028-02-29", "Year2", "next block", None),         # outside the file, leap day: no crash
        ("2028-02-29", "Year2", "year 1 block c", BlockTarget("Year1", ("BlockC",))),
    ],
)
def test_block_taken_from_the_date_when_the_context_has_none(today, year, ref, expected):
    assert resolve_block(ref, D(today), year, "") == expected


def test_no_year_in_the_context():
    assert resolve_block("my block", TODAY, "", "") is None
    assert resolve_block("year 1 block c", TODAY, "", "") == BlockTarget("Year1", ("BlockC",))
    assert resolve_block("block c", TODAY, "", "") is None   # every year has a Block C


# ---------------------------------------------------------------- the two together


@pytest.mark.parametrize(
    "today,year,block,block_ref,time_ref,expected",
    [
        ("2026-11-26", "Year2", "BlockB", "next block", "week 2", _r("2027-02-15", "2027-02-19")),  # skips spring
        ("2026-10-01", "Year2", "BlockA", "next block", "week 1", _r("2026-11-09", "2026-11-13")),
        ("2026-11-26", "Year2", "BlockB", "last block", "week 8", _r("2026-10-26", "2026-10-30")),
        ("2027-03-01", "Year3", "Semester2", "last block", "week 12", _r("2026-11-23", "2026-11-27")),
        ("2026-11-26", "Year2", "BlockB", "year 1 block d", "week 10", _r("2027-06-21", "2027-06-25")),
        ("2027-06-21", "Year1", "BlockD", "next block", "week 1", None),  # next academic year not in the file
        ("2026-11-26", "Year3", "Semester2", "next block", "week 1", None),
    ],
)
def test_block_then_week(today, year, block, block_ref, time_ref, expected):
    target = resolve_block(block_ref, D(today), year, block)
    assert resolve_time(time_ref, D(today), target) == expected
