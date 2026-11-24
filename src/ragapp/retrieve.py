"""Query time, step 1: find the chunks most similar to the question."""

import re

from ragapp import config
from ragapp.embed import embed
from ragapp.store import get_client, search

ORDINALS = {"first": "1", "second": "2", "third": "3"}


def year_block_in_question(question: str) -> tuple[str, list[str]]:
    """Find the year and ALL blocks the question names, in the site's folder format.

    'Year 2 Block B ...'        -> ('Year2', ['BlockB'])
    'Block A and Block B ...'   -> ('', ['BlockA', 'BlockB'])   (a question can name several blocks)
    'first-year ...'            -> ('Year1', [])
    'Y2B'                       -> ('Year2', ['BlockB'])
    """
    q = question.lower()
    year = ""
    blocks = set()

    if m := re.search(r"\by([123])([a-d])\b", q):  # Y2B
        year = m.group(1)
        blocks.add(f"Block{m.group(2).upper()}")
    elif m := re.search(r"\byear\s*([123])\b", q):  # Year 2
        year = m.group(1)
    elif m := re.search(r"\b(first|second|third)[\s-]year\b", q):  # second-year
        year = ORDINALS[m.group(1)]

    blocks.update(f"Block{letter.upper()}" for letter in re.findall(r"\bblock\s*([a-d])\b", q))
    blocks.update(f"Semester{n}" for n in re.findall(r"\bsemester\s*([12])\b", q))
    return (f"Year{year}" if year else "", sorted(blocks))


def retrieve(question: str, k: int = config.TOP_K, user_year: str | None = None) -> list[dict]:
    """user_year: the student's own year (e.g. 'Year2'), used when the question names no year."""
    year, blocks = year_block_in_question(question) if config.FILTER_BY_QUESTION_YEAR else ("", [])
    if not year and user_year:
        year = user_year
    # A block letter without a year is ambiguous (every year has a Block B), so it needs a year.
    query_vector = embed([question])[0]
    return search(get_client(), query_vector, k, year=year or None, blocks=blocks if year else None)
