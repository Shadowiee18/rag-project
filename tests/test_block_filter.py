"""Phase 3 step 4: the search filter, the week phrase and the student line come from the resolver and the
calendar, for every year and block (before: the student's own block, and a week read from one page format).

No Ollama, Qdrant or course data needed.
"""

from datetime import date

import pytest

import ragapp.rewrite as rewrite
import ragapp.student as student
from ragapp import config
from ragapp.retrieve import filter_for
from ragapp.student import StudentContext, describe, search_phrase

TODAY = date(2026, 11, 26)  # Year 2 Block B week 3 on the calendar

SEGMENTS = [("Year1", "BlockA"), ("Year1", "BlockB"), ("Year1", "BlockC"), ("Year1", "BlockD"),
            ("Year2", "BlockA"), ("Year2", "BlockB"), ("Year2", "BlockC"), ("Year2", "BlockD"),
            ("Year3", "Semester1"), ("Year3", "Semester2")]


# ---------------------------------------------------------------- filter: relative blocks


@pytest.mark.parametrize(
    "year,block,expected",
    [
        ("Year1", "BlockA", ("Year1", ["BlockB"])),
        ("Year1", "BlockB", ("Year1", ["BlockC"])),
        ("Year1", "BlockC", ("Year1", ["BlockD"])),
        ("Year1", "BlockD", ("Year2", ["BlockA"])),
        ("Year2", "BlockA", ("Year2", ["BlockB"])),
        ("Year2", "BlockB", ("Year2", ["BlockC"])),
        ("Year2", "BlockC", ("Year2", ["BlockD"])),
        ("Year2", "BlockD", ("Year3", ["Semester1"])),
        ("Year3", "Semester1", ("Year3", ["Semester2"])),
        ("Year3", "Semester2", ("Year3", None)),  # nothing after it: the whole year, not the own block
    ],
)
def test_next_block_question_filters_on_the_next_block(year, block, expected):
    assert filter_for("what do we have next block", year, block, TODAY) == expected


@pytest.mark.parametrize(
    "year,block,expected",
    [
        ("Year1", "BlockA", ("Year1", None)),
        ("Year1", "BlockB", ("Year1", ["BlockA"])),
        ("Year2", "BlockA", ("Year1", ["BlockD"])),
        ("Year2", "BlockB", ("Year2", ["BlockA"])),
        ("Year3", "Semester1", ("Year2", ["BlockD"])),
        ("Year3", "Semester2", ("Year3", ["Semester1"])),
    ],
)
def test_last_block_question_filters_on_the_previous_block(year, block, expected):
    assert filter_for("What did we do in the last block?", year, block, TODAY) == expected


@pytest.mark.parametrize("year,block", SEGMENTS)
def test_directional_next_block_changes_the_filter(year, block):
    """CheckList-style DIR test: asking about the next block must not search the student's own block."""
    own = filter_for("what do we have this block", year, block, TODAY)
    assert own == (year, [block])
    assert filter_for("what do we have next block", year, block, TODAY) != own


def test_previous_block_spelling():
    assert filter_for("topics of the previous block", "Year2", "BlockB", TODAY) == ("Year2", ["BlockA"])


def test_next_block_without_a_block_in_the_context_takes_it_from_the_date():
    assert filter_for("what is next block about", "Year2", None, TODAY) == ("Year2", ["BlockC"])


def test_next_block_needs_a_year():
    assert filter_for("what is next block about", None, None, TODAY) == (None, None)


def test_next_block_is_ignored_when_question_filters_are_off(monkeypatch):
    monkeypatch.setattr(config, "FILTER_BY_QUESTION_YEAR", False)
    assert filter_for("what do we have next block", "Year2", "BlockB", TODAY) == ("Year2", ["BlockB"])


def test_a_named_block_wins_over_next_block():
    assert filter_for("after year 1 block c, what is next block", "Year2", "BlockB", TODAY) == ("Year1", ["BlockC"])


# ---------------------------------------------------------------- filter: named blocks, unchanged and normalised


@pytest.mark.parametrize(
    "question,year,block,expected",
    [
        ("whats due this week", "Year2", "BlockB", ("Year2", ["BlockB"])),          # the student's own
        ("whats due this week", "Year2", None, ("Year2", None)),                    # no block known: whole year
        ("whats due this week", None, None, (None, None)),
        ("Year 1 Block C deadline", "Year2", "BlockB", ("Year1", ["BlockC"])),
        ("Block C final submission", "Year2", "BlockB", ("Year2", ["BlockC"])),     # own year
        ("Block C final submission", None, None, (None, None)),                     # every year has a Block C
        ("first-year rules", "Year2", "BlockB", ("Year1", [])),
        ("Year 2 Block A and Year 2 Block B textbooks", None, None, ("Year2", ["BlockA", "BlockB"])),
        # Year 3 pages sit in Semester folders; Year 1-2 pages in Block folders
        ("block c deliverables", "Year3", "Semester1", ("Year3", ["Semester2"])),
        ("Year 3 block a grading", "Year2", "BlockB", ("Year3", ["Semester1"])),
        ("semester 1 grades", "Year2", "BlockC", ("Year2", ["BlockA", "BlockB"])),
        ("Year 3 semester 1 grades", None, None, ("Year3", ["Semester1"])),
        ("whats due this week", "Year3", "BlockC", ("Year3", ["Semester2"])),       # Year 3 context as a letter
    ],
)
def test_named_and_own_blocks(question, year, block, expected):
    assert filter_for(question, year, block, TODAY) == expected


# ---------------------------------------------------------------- week phrase for time questions


@pytest.mark.parametrize(
    "today,year,block,expected",
    [
        ("2026-09-01", "Year1", "BlockA", "Year 1 Block A week 1"),
        ("2026-10-28", "Year1", "BlockA", "Year 1 Block A week 8"),   # after the autumn holiday
        ("2027-01-05", "Year1", "BlockB", "Year 1 Block B week 7"),   # after Christmas
        ("2027-03-01", "Year1", "BlockC", "Year 1 Block C week 4"),
        ("2027-05-04", "Year1", "BlockD", "Year 1 Block D week 3"),   # after the May holiday
        ("2026-11-02", "Year2", "BlockA", "Year 2 Block A week 9"),
        ("2026-11-24", "Year2", "BlockB", "Year 2 Block B week 3"),
        ("2027-02-16", "Year2", "BlockC", "Year 2 Block C week 2"),   # after the spring holiday
        ("2027-06-22", "Year2", "BlockD", "Year 2 Block D week 10"),
        ("2026-11-24", "Year3", "Semester1", "Year 3 Semester 1 week 12"),
        ("2027-02-02", "Year3", "Semester2", "Year 3 Semester 2 week 1"),
        ("2026-12-22", "Year2", "BlockB", "Year 2 Block B"),          # holiday: no week
        ("2027-01-26", "Year2", "BlockB", "Year 2 Block B"),          # study week: no week
        ("2027-03-01", "Year2", "BlockB", "Year 2 Block B"),          # context says B, the date is in C
        ("2028-02-29", "Year2", "BlockB", "Year 2 Block B"),          # outside the calendar, leap day
        ("2026-11-24", "Year2", "", "Year 2"),
        ("2026-11-24", "", "", ""),
    ],
)
def test_search_phrase_week_comes_from_the_calendar(today, year, block, expected):
    assert search_phrase(StudentContext(date.fromisoformat(today), year, block)) == expected


def _no_expansion(monkeypatch):
    monkeypatch.setattr(rewrite, "expand_query", lambda q: q)


def test_next_block_question_gets_no_week_phrase_of_the_current_block(monkeypatch):
    _no_expansion(monkeypatch)
    ctx = StudentContext(TODAY, "Year2", "BlockB")
    assert rewrite.search_queries("what do we have next block", ctx) == ["what do we have next block"]
    assert rewrite.week_queries("what do we have next block", ctx) == ["what do we have next block"]


def test_time_question_still_gets_the_calendar_week(monkeypatch):
    _no_expansion(monkeypatch)
    ctx = StudentContext(date(2027, 3, 1), "Year1", "BlockC")
    assert rewrite.search_queries("whats due this week", ctx) == ["Year 1 Block C week 4 whats due this week"]


# ---------------------------------------------------------------- the student line in the answer prompt


@pytest.mark.parametrize(
    "today,year,block,expected",
    [
        ("2027-03-01", "Year1", "BlockC",
         "Today is Monday 1 March 2027. The student is in Year 1, Block C, week 4 (01.03–05.03)."),
        ("2026-11-26", "Year3", "Semester1",
         "Today is Thursday 26 November 2026. The student is in Year 3, Semester 1, week 12 (23.11–27.11)."),
        ("2026-12-22", "Year2", "BlockB",
         "Today is Tuesday 22 December 2026. The student is in Year 2, Block B. This week is the Christmas holiday."),
        ("2027-01-26", "Year2", "BlockB",
         "Today is Tuesday 26 January 2027. The student is in Year 2, Block B. "
         "This week is the study week (no teaching on the calendar)."),
        ("2027-03-26", "Year2", "BlockC",
         "Today is Friday 26 March 2027. The student is in Year 2, Block C, week 7 (22.03–26.03). "
         "BUas is closed today (Good Friday)."),
        ("2027-03-01", "Year2", "BlockB",
         "Today is Monday 1 March 2027. The student is in Year 2, Block B."),       # context and date disagree
        ("2026-11-26", "Year2", "",
         "Today is Thursday 26 November 2026. The student is in Year 2."),
        ("2026-11-26", "", "", "Today is Thursday 26 November 2026."),
    ],
)
def test_describe_uses_the_calendar(today, year, block, expected):
    assert describe(StudentContext(date.fromisoformat(today), year, block), with_plan=False) == expected


TIMELINE = """Week | Self-Study | Project Task
W3 23.11–27.11 | Topic Alpha | Tasks 1-2
W4 30.11–04.12 | Topic Beta | Task 3
"""


def test_describe_adds_the_page_plan_when_the_page_has_a_dated_row(monkeypatch):
    monkeypatch.setattr(student, "_pages", lambda: {"Year2/BlockB": TIMELINE})
    ctx = StudentContext(TODAY, "Year2", "BlockB")
    assert describe(ctx) == (
        "Today is Thursday 26 November 2026. The student is in Year 2, Block B, week 3 (23.11–27.11): "
        "Topic Alpha | Tasks 1-2."
    )


def test_describe_skips_a_page_row_whose_week_disagrees_with_the_calendar(monkeypatch):
    monkeypatch.setattr(student, "_pages", lambda: {"Year2/BlockB": TIMELINE.replace("W3 ", "W5 ")})
    assert describe(StudentContext(TODAY, "Year2", "BlockB")).endswith("week 3 (23.11–27.11).")


def test_describe_without_a_page_row_still_gives_the_week(monkeypatch):
    monkeypatch.setattr(student, "_pages", lambda: {})
    assert describe(StudentContext(date(2027, 3, 1), "Year1", "BlockC")).endswith("week 4 (01.03–05.03).")
