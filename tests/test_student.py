from datetime import date

import ragapp.rewrite as rewrite
from ragapp.generate import build_prompt
from ragapp.student import find_week, nearest_year

TIMELINE = """## Project Timeline

Week | Self-Study | DataLab Task
W3 23.11–27.11 | Computer Vision | Tasks 4-5
W6 14.12–18.12 | Robotics & RL | All Tasks
x-mas |  |
W7 04.01–08.01 | N.A. | All Tasks
"""


def test_nearest_year_handles_new_year():
    assert nearest_year(4, 1, date(2026, 12, 20)) == date(2027, 1, 4)
    assert nearest_year(14, 12, date(2027, 1, 5)) == date(2026, 12, 14)


def test_find_week_inside_a_week():
    week = find_week(TIMELINE, date(2026, 11, 24))
    assert week["week"] == 3
    assert week["plan"] == "Computer Vision | Tasks 4-5"


def test_find_week_after_new_year():
    assert find_week(TIMELINE, date(2027, 1, 6))["week"] == 7


def test_find_week_in_holiday_gap_is_none():
    assert find_week(TIMELINE, date(2026, 12, 28)) is None


def test_build_prompt_adds_student_line_before_question():
    prompt = build_prompt("whats due", [{"title": "T", "text": "x"}], student="Today is Tuesday 24 November 2026.")
    assert prompt.endswith("Student: Today is Tuesday 24 November 2026.\n\nQuestion: whats due")


class _FakeResponse:
    class message:
        content = "some query"


def test_rewrite_drops_student_context_when_question_names_a_block(monkeypatch):
    """The model once rewrote 'year 2 block a' into the student's own Block B. The code prevents that."""
    seen = {}
    monkeypatch.setattr(rewrite.ollama, "chat", lambda **kw: seen.update(kw) or _FakeResponse())

    rewrite.rewrite_query("deadline data quality report year 2 block a", "The student is in Year 2, Block B.")
    assert "Block B" not in seen["messages"][0]["content"]
    assert "never add a year, block or week" in seen["messages"][0]["content"]

    rewrite.rewrite_query("whats due this week", "The student is in Year 2, Block B.")
    assert "The student is in Year 2, Block B." in seen["messages"][0]["content"]
