"""The behavioural suite (Phase 3 step 3): template questions over every segment and date position, with gold
answers computed from calendar.yaml by the suite's own code, scored by exact match. No Ollama/Qdrant needed."""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import behavioral as bh  # noqa: E402

CASES = bh.make_cases()
SEGMENTS = {f"{y} {b}" for y, b in bh.SEGMENTS}


def _find(**fields):
    return [c for c in CASES if all(c[k] == v for k, v in fields.items())]


# ---------------------------------------------------------------- the cases


def test_every_family_covers_every_segment_it_can():
    by_family = {f: {c["segment"] for c in CASES if c["family"] == f} for f in bh.FAMILIES}
    assert by_family["week"] == SEGMENTS
    assert by_family["week_dates"] == SEGMENTS
    assert by_family["dir_next"] == SEGMENTS - {"Year3 Semester2"}   # nothing after Year 3
    assert by_family["dir_last"] == SEGMENTS - {"Year1 BlockA"}      # nothing before Year 1
    # the next block must be in this academic year's calendar: not after Block D / Semester 2
    assert by_family["next_start"] == SEGMENTS - {"Year1 BlockD", "Year2 BlockD", "Year3 Semester2"}


def test_size_and_unique_ids():
    assert len(CASES) >= 120
    assert len({c["id"] for c in CASES}) == len(CASES)


def test_week_cases_come_in_reworded_pairs():
    groups = Counter(c["inv_group"] for c in CASES if c["family"] == "week")
    assert set(groups.values()) == {2}


@pytest.mark.parametrize(
    "segment,today,gold",
    [
        ("Year2 BlockB", "2026-11-24", {"week": 3}),
        ("Year1 BlockA", "2026-09-01", {"week": 1}),
        ("Year1 BlockA", "2026-10-21", {"break": "autumn holiday"}),
        ("Year2 BlockB", "2026-12-23", {"break": "Christmas holiday"}),
        ("Year2 BlockB", "2027-01-27", {"break": "study week (no teaching on the calendar)"}),
        ("Year2 BlockB", "2027-01-21", {"week": 9}),                 # Thursday of the last teaching week
        ("Year1 BlockD", "2027-06-24", {"week": 10}),
        ("Year3 Semester1", "2027-01-21", {"week": 18}),             # semester weeks run on through Block B
        ("Year3 Semester2", "2027-02-02", {"week": 1}),
        ("Year3 Semester2", "2027-04-28", {"break": "May holiday"}),
    ],
)
def test_week_gold(segment, today, gold):
    cases = _find(family="week", segment=segment, today=today)
    assert len(cases) == 2 and all(c["gold"] == gold for c in cases)


@pytest.mark.parametrize(
    "segment,week,gold",
    [
        ("Year2 BlockB", 2, ["2026-11-16", "2026-11-20"]),
        ("Year2 BlockB", 9, ["2027-01-18", "2027-01-22"]),
        ("Year1 BlockC", 2, ["2027-02-15", "2027-02-19"]),           # after the spring holiday
        ("Year2 BlockD", 10, ["2027-06-21", "2027-06-25"]),
        ("Year3 Semester1", 18, ["2027-01-18", "2027-01-22"]),
        ("Year3 Semester2", 19, ["2027-06-21", "2027-06-25"]),
    ],
)
def test_week_dates_gold(segment, week, gold):
    [case] = [c for c in _find(family="week_dates", segment=segment) if f"week {week}?" in c["question"]]
    assert case["gold"] == {"dates": gold}


@pytest.mark.parametrize(
    "segment,gold",
    [("Year1 BlockA", "2026-11-09"), ("Year2 BlockB", "2027-02-01"), ("Year2 BlockC", "2027-04-12"),
     ("Year3 Semester1", "2027-02-01")],
)
def test_next_start_gold(segment, gold):
    [case] = _find(family="next_start", segment=segment)
    assert case["gold"] == {"date": gold}


@pytest.mark.parametrize(
    "family,segment,target",
    [("dir_next", "Year1 BlockD", ["Year2", "BlockA"]), ("dir_next", "Year2 BlockD", ["Year3", "Semester1"]),
     ("dir_next", "Year2 BlockB", ["Year2", "BlockC"]), ("dir_last", "Year2 BlockA", ["Year1", "BlockD"]),
     ("dir_last", "Year3 Semester1", ["Year2", "BlockD"]), ("dir_last", "Year3 Semester2", ["Year3", "Semester1"])],
)
def test_directional_gold(family, segment, target):
    [case] = _find(family=family, segment=segment)
    assert case["gold"] == {"target": target}


def test_every_case_date_lies_inside_its_segment():
    for c in CASES:
        assert bh.segment_span(c["year"], c["block"])[0] <= bh.D(c["today"]) <= bh.segment_span(c["year"], c["block"])[1]


def test_gold_does_not_come_from_the_apps_resolver():
    source = (Path(bh.__file__)).read_text(encoding="utf-8")
    assert "ragapp.resolver" not in source and "ragapp.student" not in source.split("def main")[0]


# ---------------------------------------------------------------- reading answers


@pytest.mark.parametrize(
    "text,weeks",
    [
        ("You are in week 3 of Block B.", [3]),
        ("This is Week 12 (W12) of the semester; week 13 starts Monday.", [12, 12, 13]),
        ("It is W5.", [5]),
        ("No week number here, 3 tasks left.", []),
    ],
)
def test_extract_weeks(text, weeks):
    assert bh.extract_weeks(text) == weeks


@pytest.mark.parametrize(
    "text,dates",
    [
        ("Week 3 runs from 23.11 to 27.11.", [(23, 11), (27, 11)]),
        ("from 23.11.2026 until 27/11/2026", [(23, 11), (27, 11)]),
        ("Monday 23 November to Friday 27 November 2026", [(23, 11), (27, 11)]),
        ("23–27 November", [(23, 11), (27, 11)]),
        ("23-27 Nov", [(23, 11), (27, 11)]),
        ("November 23rd to November 27th", [(23, 11), (27, 11)]),
        ("It starts on 2027-02-01.", [(1, 2)]),
        ("Block C starts on 1 February 2027 [2].", [(1, 2)]),
        ("Task 3 is due at 17:00.", []),
        ("Version 2.0 of the rules", []),
    ],
)
def test_extract_dates(text, dates):
    assert bh.extract_dates(text) == dates


@pytest.mark.parametrize(
    "gold,answer,passed",
    [
        ({"week": 3}, "Today you are in week 3 (23.11–27.11) of Block B.", True),
        ({"week": 3}, "You are in week 4.", False),
        ({"week": 3}, "I don't know which week it is.", False),
        ({"week": 3}, "Week 3; next week is week 4.", True),          # the first week named counts
        ({"break": "Christmas holiday"}, "There is no teaching: it is the Christmas break.", True),
        ({"break": "Christmas holiday"}, "You are in week 7.", False),
        ({"break": "study week"}, "This is the study week, no classes are scheduled.", True),
        ({"dates": ["2026-11-16", "2026-11-20"]}, "Week 2 is 16.11–20.11.", True),
        ({"dates": ["2026-11-16", "2026-11-20"]}, "Week 2 is 16-20 November 2026.", True),
        ({"dates": ["2026-11-16", "2026-11-20"]}, "Week 2 starts on 16 November.", False),  # no end date
        ({"dates": ["2026-11-16", "2026-11-20"]}, "Week 2 is 9.11–13.11.", False),
        ({"date": "2027-02-01"}, "The next block starts on Monday 1 February 2027.", True),
        ({"date": "2027-02-01"}, "It starts on 8 February.", False),
        ({"date": "2027-02-01"}, "I don't know.", False),
    ],
)
def test_score_answers(gold, answer, passed):
    assert bh.score_answer(gold, answer) is passed


def _chunk(year, block):
    return {"doc_id": f"{year}/{block}/page", "year": year, "block": block}


@pytest.mark.parametrize(
    "chunks,passed",
    [
        ([_chunk("Year2", "BlockC"), _chunk("Year2", "BlockB")], True),
        ([_chunk("Year2", "BlockB"), _chunk("Year2", "BlockC")], False),   # top result from the own block
        ([_chunk("", ""), _chunk("Year2", "BlockC")], False),               # a shared page on top
        ([{"doc_id": "x", "year": "Year2", "block": ["BlockB", "BlockC"]}], True),  # merged copy of both
        ([], False),
    ],
)
def test_score_directional(chunks, passed):
    assert bh.score_retrieval(["Year2", "BlockC"], chunks) is passed


# ---------------------------------------------------------------- running and summarising


def _fake_pipeline(answers):
    def answer_fn(case):
        return {"answer": answers(case), "chunks": [_chunk(*case["gold"]["target"])] if "target" in case["gold"] else []}
    return answer_fn


def test_run_with_a_perfect_fake_pipeline_scores_everything():
    def perfect(case):
        g = case["gold"]
        if "week" in g:
            return f"You are in week {g['week']}."
        if "break" in g:
            return f"It is the {g['break']}."
        if "dates" in g:
            return f"From {bh.D(g['dates'][0]):%d.%m} to {bh.D(g['dates'][1]):%d.%m}."
        if "date" in g:
            return f"On {bh.D(g['date']):%d.%m.%Y}."
        return "Here is what the next block covers."

    results = bh.run(CASES, _fake_pipeline(perfect))
    summary = bh.summarise(results)
    assert summary["passed"] == 1.0
    assert summary["invariance"]["consistent"] == 1.0
    assert set(summary["per_segment"]) == SEGMENTS


def test_run_with_a_week_3_parrot():
    """A pipeline that always answers 'week 3' passes only where week 3 is right, and is perfectly consistent."""
    week_cases = _find(family="week")
    results = bh.run(week_cases, _fake_pipeline(lambda case: "You are in week 3."))
    summary = bh.summarise(results)
    right = sum(c["gold"] == {"week": 3} for c in week_cases)
    assert 0 < right < len(week_cases) / 4
    assert summary["per_family"]["week"] == round(right / len(week_cases), 3)
    assert summary["invariance"]["consistent"] == 1.0
    assert summary["per_segment"]["Year2 BlockB"] > 0  # 24 Nov is week 3


def test_inconsistent_rewording_is_caught():
    pair = [c for c in CASES if c["family"] == "week"][:2]
    results = bh.run(pair, _fake_pipeline(lambda case: "week 1" if case["id"].endswith("-a") else "week 2"))
    assert bh.summarise(results)["invariance"]["consistent"] == 0.0
