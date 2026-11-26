"""The student profile rules: the study year is asked once per academic year; block and week come from the date."""

from datetime import date

import pytest

from ragapp.profile import academic_year, needs_year, status, student_context
from ragapp.student import StudentContext, describe

D = date.fromisoformat


@pytest.mark.parametrize("today, expected", [
    ("2026-11-26", "2026-27"),  # in calendar.yaml
    ("2026-08-24", "2026-27"),  # first day the file knows (introduction week)
    ("2027-08-29", "2026-27"),  # last day of the file
    ("2027-09-15", "2027-28"),  # after the file: the Dutch academic year, 1 September - 31 August
    ("2026-05-01", "2025-26"),  # before the file
    ("2028-03-01", "2027-28"),
])
def test_academic_year(today, expected):
    assert academic_year(D(today)) == expected


@pytest.mark.parametrize("year, given_in, today, expected", [
    ("", "", "2026-11-26", True),               # never asked
    ("Year2", "2026-27", "2026-11-26", False),  # asked this academic year
    ("Year2", "2026-27", "2027-07-20", False),  # summer still belongs to 2026-27
    ("Year2", "2026-27", "2027-09-15", True),   # a new academic year: ask again
    ("Year2", "2025-26", "2026-11-26", True),
    ("Year4", "2026-27", "2026-11-26", True),   # not a study year
])
def test_needs_year(year, given_in, today, expected):
    assert needs_year(year, given_in, D(today)) is expected


@pytest.mark.parametrize("year, today, block", [
    ("Year2", "2026-11-26", "BlockB"),
    ("Year1", "2027-03-01", "BlockC"),
    ("Year3", "2026-11-26", "Semester1"),
    ("Year2", "2026-12-23", "BlockB"),  # the Christmas holiday is inside Block B
    ("Year2", "2027-07-20", ""),        # summer holiday: no block
    ("Year2", "2028-03-01", ""),        # not in the calendar file: no block, not a guess
])
def test_student_context_takes_the_block_from_the_date(year, today, block):
    assert student_context(year, D(today)) == StudentContext(D(today), year, block)


def test_student_context_block_override_wins():
    assert student_context("Year2", D("2026-11-26"), "BlockC").block == "BlockC"


@pytest.mark.parametrize("year, today, block, label", [
    ("Year2", "2026-11-26", "", "Year 2 · Block B · week 3"),
    ("Year3", "2026-11-26", "", "Year 3 · Semester 1 · week 12"),
    ("Year2", "2026-12-23", "", "Year 2 · Block B · Christmas holiday"),
    ("Year2", "2027-07-20", "", "Year 2 · block unknown (summer holiday)"),
    ("Year2", "2028-03-01", "", "Year 2 · block unknown (the date is not in the academic calendar)"),
    ("Year2", "2026-11-26", "BlockC", "Year 2 · Block C"),  # an override outside its dates has no week
])
def test_status_label(year, today, block, label):
    s = status(year, academic_year(D(today)), D(today), block)
    assert s["ask_year"] is False
    assert s["label"] == label


def test_status_of_a_current_profile():
    assert status("Year2", "2026-27", D("2026-11-26")) == {
        "ask_year": False, "academic_year": "2026-27", "year": "Year2", "block": "BlockB", "week": 3,
        "label": "Year 2 · Block B · week 3",
    }


def test_status_of_a_stale_profile_asks_again_and_keeps_the_old_year():
    assert status("Year2", "2026-27", D("2027-09-15")) == {
        "ask_year": True, "academic_year": "2027-28", "year": "Year2", "block": None, "week": None, "label": "",
    }


def test_status_without_a_profile():
    assert status("", "", D("2026-11-26"))["ask_year"] is True
    assert status("", "", D("2026-11-26"))["year"] is None


def test_describe_says_when_the_block_is_unknown():
    assert describe(StudentContext(D("2027-07-20"), "Year2")) == (
        "Today is Tuesday 20 July 2027. The student is in Year 2. The block is not known today (summer holiday).")
    assert describe(StudentContext(D("2028-03-01"), "Year2")).endswith(
        "The block is not known today (the date is not in the academic calendar).")


def test_describe_without_a_block_on_a_teaching_day_is_unchanged():
    assert describe(StudentContext(D("2026-11-24"), "Year2")) == "Today is Tuesday 24 November 2026. The student is in Year 2."
