"""Query time, step 1: find the chunks most similar to the question."""

import re

from ragapp import config
from ragapp.bm25 import bm25_search
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


def filter_for(question: str, user_year: str | None, user_block: str | None) -> tuple[str | None, list[str] | None]:
    """Year/block to filter on: from the question if it names one, else the student's own.
    A block letter without a year is ambiguous (every year has a Block B), so it needs a year."""
    year, blocks = year_block_in_question(question) if config.FILTER_BY_QUESTION_YEAR else ("", [])
    if not year and user_year:
        year = user_year
        if not blocks and user_block:
            blocks = [user_block]
    return (year or None), (blocks if year else None)


def retrieve(
    question: str, k: int = config.TOP_K, user_year: str | None = None, user_block: str | None = None
) -> list[dict]:
    """user_year / user_block: the student's own year and block (e.g. 'Year2', 'BlockB'),
    used only when the question names no year itself. Pages with no year/block always stay searchable."""
    year, blocks = filter_for(question, user_year, user_block)
    query_vector = embed([question])[0]
    if not config.HYBRID:
        return search(get_client(), query_vector, k, year=year, blocks=blocks)
    depth = max(k, FUSION_DEPTH)
    dense = search(get_client(), query_vector, depth, year=year, blocks=blocks)
    return rrf_merge([dense, bm25_search(question, depth, year, blocks)], k)  # meaning + exact words


RRF_K = 60  # the usual constant: dampens how much rank 1 beats rank 2
# How deep every list fused with RRF is, whatever k is. A chunk ranked 12th in two lists beats one ranked 1st in
# one list, so a deeper list can change the TOP of the fused result. A fixed depth keeps the app's top 5 (k=5)
# equal to the first 5 of the eval's top 10 (k=10).
FUSION_DEPTH = 20


def rrf_merge(result_lists: list[list[dict]], k: int) -> list[dict]:
    """Reciprocal rank fusion: each chunk scores sum(1 / (RRF_K + rank)) over the lists it appears in."""
    scores: dict[tuple, float] = {}
    chunk_by_key: dict[tuple, dict] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            key = (chunk["doc_id"], chunk["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            chunk_by_key.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: -scores[key])[:k]
    return [{**chunk_by_key[key], "score": round(scores[key], 5)} for key in ordered]


def retrieve_many(
    queries: list[str], k: int, question: str, user_year: str | None = None, user_block: str | None = None
) -> list[dict]:
    """Search with several queries and merge with RRF. The filter is decided from the ORIGINAL question,
    so a rewrite can't change which block is searched. With HYBRID, every query also gets a BM25 search."""
    year, blocks = filter_for(question, user_year, user_block)
    client = get_client()
    vectors = embed(queries)  # one batch call for all queries
    depth = max(2 * k, FUSION_DEPTH)
    lists = [search(client, v, depth, year=year, blocks=blocks) for v in vectors]
    if config.HYBRID:
        lists += [bm25_search(q, depth, year, blocks) for q in queries]
    return rrf_merge(lists, k)
