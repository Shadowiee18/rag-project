"""Calendar facts (Phase 3 step 6): find the time and block words in a question with the resolver, and write the
matching facts from the academic calendar as short sentences the answering model can use.

Expected dates are typed in by hand from the 2026-27 academic calendar, independently of calendar.yaml.
No Ollama, Qdrant or course data needed.
"""

from datetime import date

import pytest

from ragapp.calendar_facts import Reference, calendar_facts, find_references
from ragapp.student import StudentContext

D = date.fromisoformat
Y2B = StudentContext(D("2026-11-24"), "Year2", "BlockB")  # Tuesday, Year 2 Block B week 3


def facts(question: str, ctx: StudentContext = Y2B) -> str:
    return "\n".join(calendar_facts(question, ctx))


# ---------------------------------------------------------------- finding the references in a question


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What are the dates of week 2?", [Reference("week", "week 2")]),
        ("When does the next block start?", [Reference("block", "next block")]),
        ("When does the next semester start?", [Reference("block", "next semester")]),
        ("Is there class on 8 February?", [Reference("time", "8 february")]),
        ("what do we have tomorrow", [Reference("time", "tomorrow")]),
        ("what is due next week?", [Reference("time", "next week")]),
        ("what's due in week 5 of block c", [Reference("week", "week 5"), Reference("block", "block c")]),
        ("deadline 14.12?", [Reference("time", "14.12")]),
        ("Which teaching week is the 2nd of December in?", [Reference("time", "2nd of december")]),
        ("what do Y1C students do?", [Reference("block", "y1c")]),
        ("what happens in Year 1 Block C week 4", [Reference("block", "year 1 block c"), Reference("week", "week 4")]),
        ("is wednesday a lecture day", [Reference("time", "wednesday")]),
        ("Which week of the block are we in?", []),
        ("What week is it right now?", []),
    ],
)
def test_references_are_the_resolvers_forms_found_in_the_question(question, expected):
    assert find_references(question, Y2B) == expected


@pytest.mark.parametrize(
    "question",
    [
        "what are tasks 1-2 about?",   # a range of numbers, not 1 February
        "I sat the exam, what now?",   # 'sat' the verb, not Saturday
        "is the sun module hard?",
        "who wed the data?",
    ],
)
def test_forms_that_are_ambiguous_in_running_text_are_not_references(question):
    assert find_references(question, Y2B) == []


# ---------------------------------------------------------------- today, always


def test_today_in_a_teaching_week():
    text = facts("hello")
    assert "Today, Tue 24 Nov 2026" in text
    assert "Year 2 Block B, teaching week 3" in text
    assert "Mon 23 Nov 2026" in text and "Fri 27 Nov 2026" in text


@pytest.mark.parametrize(
    "ctx,name,start,end,next_week,next_monday",
    [
        (StudentContext(D("2026-12-23"), "Year2", "BlockB"), "Christmas holiday", "Mon 21 Dec 2026", "Sun 3 Jan 2027",
         "Year 2 Block B week 7", "Mon 4 Jan 2027"),
        (StudentContext(D("2026-10-21"), "Year1", "BlockA"), "autumn holiday", "Mon 19 Oct 2026", "Sun 25 Oct 2026",
         "Year 1 Block A week 8", "Mon 26 Oct 2026"),
        (StudentContext(D("2027-01-27"), "Year3", "Semester1"), "study week", "Mon 25 Jan 2027", "Sun 31 Jan 2027",
         "Year 3 Semester 2 week 1", "Mon 1 Feb 2027"),
        (StudentContext(D("2027-04-28"), "Year2", "BlockD"), "May holiday", "Mon 26 Apr 2027", "Sun 2 May 2027",
         "Year 2 Block D week 3", "Mon 3 May 2027"),
    ],
)
def test_today_in_a_break_has_no_week_number_and_names_the_next_teaching_week(ctx, name, start, end, next_week,
                                                                               next_monday):
    today = calendar_facts("Which week is it?", ctx)[0]
    assert name in today and start in today and end in today
    assert "no teaching" in today and "no week number" in today
    assert "teaching week" not in today.split(".")[0]  # the first sentence says what today is: not a week
    assert f"Next teaching week: {next_week}, from {next_monday}" in today


def test_last_teaching_week_before_a_break():
    today = calendar_facts("hi", StudentContext(D("2026-12-23"), "Year2", "BlockB"))[0]
    assert "Last teaching week before it: Year 2 Block B week 6 (Mon 14 Dec 2026 - Fri 18 Dec 2026)" in today


def test_today_on_a_closed_day_and_at_the_weekend():
    assert "BUas is closed today (Liberation Day)" in facts("hi", StudentContext(D("2027-05-05"), "Year1", "BlockD"))
    assert "weekend" in facts("hi", StudentContext(D("2026-11-28"), "Year2", "BlockB"))


def test_today_outside_the_calendar_file():
    assert "not in the academic calendar" in facts("hi", StudentContext(D("2028-01-10"), "Year2", "BlockB"))


def test_today_without_a_year_still_says_the_block_and_week():
    assert "Block B, teaching week 3" in facts("hi", StudentContext(D("2026-11-24")))


# ---------------------------------------------------------------- the student's own block, always


def test_own_block_start_end_breaks_and_closed_days():
    text = facts("hello")
    assert "Year 2 Block B (the student's block)" in text
    assert "week 1 starts Mon 9 Nov 2026" in text
    assert "last teaching week, week 9: Mon 18 Jan 2027 - Fri 22 Jan 2027 (assessment)" in text
    assert "Christmas holiday (Mon 21 Dec 2026 - Sun 3 Jan 2027)" in text
    assert "Fri 1 Jan 2027 (New Year's Day)" in text


# ---------------------------------------------------------------- week N


@pytest.mark.parametrize(
    "ctx,question,expected",
    [
        (Y2B, "What are the dates of week 2?", "Week 2 of Year 2 Block B: Mon 16 Nov 2026 - Fri 20 Nov 2026"),
        (Y2B, "dates of week 7?", "Week 7 of Year 2 Block B: Mon 4 Jan 2027 - Fri 8 Jan 2027"),  # after Christmas
        (StudentContext(D("2027-04-20"), "Year1", "BlockD"), "week 10?",
         "Week 10 of Year 1 Block D: Mon 21 Jun 2027 - Fri 25 Jun 2027"),
        (StudentContext(D("2027-02-17"), "Year3", "Semester2"), "What are the dates of week 19?",
         "Week 19 of Year 3 Semester 2: Mon 21 Jun 2027 - Fri 25 Jun 2027"),
        (Y2B, "what's due in week 5 of block c", "Week 5 of Year 2 Block C: Mon 8 Mar 2027 - Fri 12 Mar 2027"),
    ],
)
def test_week_n_dates(ctx, question, expected):
    assert expected in facts(question, ctx)


def test_week_n_lists_closed_days_and_notes():
    text = facts("what about week 3?", StudentContext(D("2027-04-20"), "Year2", "BlockD"))
    assert "Week 3 of Year 2 Block D: Mon 3 May 2027 - Fri 7 May 2027" in text
    assert "Wed 5 May 2027 (Liberation Day)" in text and "Thu 6 May 2027 (Ascension Day)" in text
    assert "assessment" in facts("week 9?")


def test_a_week_the_block_does_not_have():
    assert "Year 2 Block B has teaching weeks 1-9; there is no week 12" in facts("What are the dates of week 12?")


# ---------------------------------------------------------------- next / last block, named blocks


@pytest.mark.parametrize(
    "ctx,question,expected",
    [
        (Y2B, "When does the next block start?", "Next block = Year 2 Block C"),
        (Y2B, "When does the next block start?", "week 1 starts Mon 1 Feb 2027"),
        (StudentContext(D("2026-09-15"), "Year1", "BlockA"), "When does the next block start?",
         "week 1 starts Mon 9 Nov 2026"),
        (StudentContext(D("2026-09-15"), "Year3", "Semester1"), "When does the next semester start?",
         "Next semester = Year 3 Semester 2"),
        (StudentContext(D("2026-09-15"), "Year3", "Semester1"), "When does the next semester start?",
         "week 1 starts Mon 1 Feb 2027"),
        (Y2B, "what did we do last block?", "Last block = Year 2 Block A"),
        (Y2B, "when does block d end?", "Year 2 Block D: teaching week 1 starts Mon 12 Apr 2027"),
        (Y2B, "when does block d end?", "the regulations give Fri 18 Jun 2027"),
        (Y2B, "when does block d end?", "Wed 5 May 2027 (Liberation Day)"),
    ],
)
def test_block_facts(ctx, question, expected):
    assert expected in facts(question, ctx)


def test_a_block_in_the_next_academic_year_is_not_in_the_calendar_yet():
    text = facts("when does the next block start?", StudentContext(D("2027-05-11"), "Year1", "BlockD"))
    assert "Next block = Year 2 Block A" in text and "not in the calendar yet" in text


def test_a_semester_of_a_block_year_describes_both_blocks():
    text = facts("when does semester 2 start?", StudentContext(D("2026-09-15"), "Year1", "BlockA"))
    assert "Year 1 Block C" in text and "Year 1 Block D" in text


def test_year3_semester_counts_weeks_across_both_blocks():
    text = facts("when does semester 2 end?", StudentContext(D("2026-09-15"), "Year3", "Semester1"))
    assert "Year 3 Semester 2: teaching week 1 starts Mon 1 Feb 2027" in text
    assert "last teaching week, week 19: Mon 21 Jun 2027 - Fri 25 Jun 2027" in text


# ---------------------------------------------------------------- days and calendar weeks


@pytest.mark.parametrize(
    "ctx,question,expected",
    [
        (Y2B, "Is there class on 8 February?", "Mon 8 Feb 2027: spring holiday, no teaching"),
        (Y2B, "Is there class on 8 February?", "BUas closed (Carnival)"),
        (StudentContext(D("2026-11-27"), "Year2", "BlockB"), "anything tomorrow?", "Tomorrow = Sat 28 Nov 2026: weekend"),
        (Y2B, "what do we have tomorrow", "Tomorrow = Wed 25 Nov 2026: Year 2 Block B teaching week 3"),
        (Y2B, "what is due next week?", "Next week = Mon 30 Nov 2026 - Fri 4 Dec 2026: Year 2 Block B teaching week 4"),
        (StudentContext(D("2026-12-15"), "Year2", "BlockB"), "is next week a holiday?",
         "Next week = Mon 21 Dec 2026 - Fri 25 Dec 2026: Christmas holiday, no teaching"),
        (Y2B, "Which teaching week is the 2nd of December in?", "Wed 2 Dec 2026: Year 2 Block B teaching week 4"),
        (StudentContext(D("2027-04-20"), "Year1", "BlockD"), "is 5 may a day off?",
         "Wed 5 May 2027: BUas closed (Liberation Day), in Year 1 Block D teaching week 3"),
    ],
)
def test_day_and_week_facts(ctx, question, expected):
    assert expected in facts(question, ctx)


def test_each_fact_once():
    found = calendar_facts("this block, my block, the current block: week 2 and week 2?", Y2B)
    assert len(found) == len(set(found))
