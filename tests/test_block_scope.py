"""BLOCK_ONLY_SEARCH: a question about one block ("what do we have next block?", "Year 1 Block C overview")
searches only that block's own pages, and its name is added to the search queries.

Before: the filter always kept the shared pages (no year/block) next to the block, and for a vague question those
ranked first. No Ollama, Qdrant or course data needed.
"""

from datetime import date

import pytest

import ragapp.retrieve as retrieve
import ragapp.rewrite as rewrite
from ragapp import config
from ragapp.retrieve import block_only_target
from ragapp.store import build_filter
from ragapp.student import StudentContext, readable_name

TODAY = date(2026, 11, 26)  # Year 2 Block B week 3 on the calendar
Y2B = StudentContext(TODAY, "Year2", "BlockB")


@pytest.fixture
def block_only(monkeypatch):
    monkeypatch.setattr(config, "BLOCK_ONLY_SEARCH", True)


def matches(f) -> dict[str, list[str]]:
    return {c.key: list(c.match.any) for c in f.must}


# ---------------------------------------------------------------- the filter


def test_default_filter_keeps_the_shared_pages():
    assert matches(build_filter("Year2", ["BlockC"])) == {"year": ["Year2", ""], "block": ["BlockC", ""]}


def test_block_only_filter_leaves_the_shared_pages_out():
    assert matches(build_filter("Year2", ["BlockC"], shared=False)) == {"year": ["Year2"], "block": ["BlockC"]}


def test_block_only_filter_still_leaves_templates_out(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_TEMPLATES", True)
    assert [c.key for c in build_filter("Year2", ["BlockC"], shared=False).must_not] == ["page_type"]


# ---------------------------------------------------------------- which questions are about one block


@pytest.mark.parametrize(
    "question,year,block,expected",
    [
        ("what do we have next block?", "Year2", "BlockB", ("Year2", ["BlockC"])),
        ("What did we do last block?", "Year2", "BlockA", ("Year1", ["BlockD"])),
        ("what is the previous block about", "Year3", "Semester1", ("Year2", ["BlockD"])),
        ("Year 1 Block C overview", None, None, ("Year1", ["BlockC"])),  # named in full: no student needed
        ("Year 1 Block C overview", "Year2", "BlockB", ("Year1", ["BlockC"])),
        ("block c overview", "Year2", "BlockB", ("Year2", ["BlockC"])),  # a letter alone: the student's year
        ("block c overview", "Year3", "Semester1", ("Year3", ["Semester2"])),  # Year 3 uses semesters
        ("Y1A deliverables", None, None, ("Year1", ["BlockA"])),
        ("Year 1 semester 1 overview", None, None, ("Year1", ["BlockA", "BlockB"])),
        ("textbooks in Year 2 Block A and Year 2 Block B", None, None, ("Year2", ["BlockA", "BlockB"])),
    ],
)
def test_questions_about_one_block(block_only, question, year, block, expected):
    assert block_only_target(question, year, block, TODAY) == expected


@pytest.mark.parametrize(
    "question,year,block",
    [
        ("whats due this week", "Year2", "BlockB"),  # no block reference: today's behaviour
        ("what is overfitting", "Year2", "BlockB"),
        ("I'm a first-year student, how late can I be?", None, None),  # a whole year, no block
        ("Year 1 overview", "Year2", "BlockB"),
        ("block c overview", None, None),  # every year has a Block C
        ("what do we have next block", None, None),  # next from what?
        ("what do we have next block", "Year3", "Semester2"),  # nothing after it: the whole year
        ("what did we do last block", "Year1", "BlockA"),  # nothing before it
    ],
)
def test_questions_not_about_one_block(block_only, question, year, block):
    assert block_only_target(question, year, block, TODAY) is None


def test_switch_off_by_default_means_no_block_only_search():
    assert config.BLOCK_ONLY_SEARCH is False
    assert block_only_target("what do we have next block?", "Year2", "BlockB", TODAY) is None


def test_no_block_only_search_when_question_filtering_is_off(block_only, monkeypatch):
    monkeypatch.setattr(config, "FILTER_BY_QUESTION_YEAR", False)
    assert block_only_target("Year 1 Block C overview", "Year2", "BlockB", TODAY) is None


# ---------------------------------------------------------------- the searches use it


@pytest.fixture
def recorded_searches(monkeypatch):
    """Replace the embedder and both searches; record the filter each search got."""
    calls = []
    monkeypatch.setattr(retrieve, "embed", lambda texts: [[0.0] for _ in texts])
    monkeypatch.setattr(retrieve, "get_client", lambda: None)
    monkeypatch.setattr(
        retrieve, "search", lambda client, v, k, year=None, blocks=None, shared=True: calls.append(("dense", year, blocks, shared)) or []
    )
    monkeypatch.setattr(
        retrieve, "bm25_search", lambda q, k, year=None, blocks=None, shared=True: calls.append(("bm25", year, blocks, shared)) or []
    )
    monkeypatch.setattr(config, "HYBRID", True)
    return calls


def test_retrieve_many_searches_only_the_target_block(block_only, recorded_searches):
    retrieve.retrieve_many(["q1", "q2"], 5, "what do we have next block?", "Year2", "BlockB", TODAY)
    assert recorded_searches and all(c[1:] == ("Year2", ["BlockC"], False) for c in recorded_searches)


def test_retrieve_searches_only_the_target_block(block_only, recorded_searches):
    retrieve.retrieve("Year 1 Block C overview", 5, "Year2", "BlockB", TODAY)
    assert recorded_searches and all(c[1:] == ("Year1", ["BlockC"], False) for c in recorded_searches)


def test_other_questions_keep_the_shared_pages(block_only, recorded_searches):
    retrieve.retrieve_many(["whats due this week"], 5, "whats due this week", "Year2", "BlockB", TODAY)
    assert all(c[1:] == ("Year2", ["BlockB"], True) for c in recorded_searches)


def test_switch_off_keeps_the_shared_pages(recorded_searches):
    retrieve.retrieve_many(["q"], 5, "what do we have next block?", "Year2", "BlockB", TODAY)
    assert all(c[1:] == ("Year2", ["BlockC"], True) for c in recorded_searches)


# ---------------------------------------------------------------- the block's name in the search queries


@pytest.mark.parametrize(
    "year,blocks,expected",
    [
        ("Year2", ["BlockC"], "Year 2 Block C"),
        ("Year3", ["Semester1"], "Year 3 Semester 1"),
        ("Year1", ["BlockA", "BlockB"], "Year 1 Block A and Block B"),
    ],
)
def test_readable_name(year, blocks, expected):
    assert readable_name(year, blocks) == expected


def test_week_queries_add_a_copy_with_the_target_block(block_only):
    assert rewrite.week_queries("what do we have next block?", Y2B) == [
        "what do we have next block?",
        "Year 2 Block C what do we have next block?",
    ]


def test_search_queries_tag_both_queries_with_the_target_block(block_only, monkeypatch):
    monkeypatch.setattr(rewrite, "expand_query", lambda q: q + " expanded")
    assert rewrite.search_queries("What did we do last block?", Y2B) == [
        "Year 2 Block A What did we do last block?",
        "Year 2 Block A What did we do last block? expanded",
    ]


def test_a_letter_alone_gets_the_full_name(block_only):
    assert rewrite.week_queries("block c overview", Y2B) == ["block c overview", "Year 2 Block C block c overview"]


def test_no_copy_when_the_question_already_says_the_name(block_only):
    assert rewrite.week_queries("Year 1 Block C overview", Y2B) == ["Year 1 Block C overview"]


def test_block_name_needs_no_student_when_the_question_names_the_year(block_only):
    assert rewrite.week_queries("y1c deliverables", None) == ["y1c deliverables", "Year 1 Block C y1c deliverables"]


def test_queries_unchanged_with_the_switch_off():
    assert rewrite.week_queries("what do we have next block?", Y2B) == ["what do we have next block?"]
