"""Query time, step 1: find the chunks most similar to the question."""

import re
from datetime import date

from ragapp import config
from ragapp.bm25 import bm25_search
from ragapp.embed import embed
from ragapp.resolver import block_ref_in_question, resolve_block, segments_for
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


def filter_for(
    question: str, user_year: str | None, user_block: str | None, today: date | None = None
) -> tuple[str | None, list[str] | None]:
    """Year/block to filter on, in this order:
    1. the year/block the question names ("Year 1 Block C"; a block letter alone takes the student's year,
       because every year has a Block B);
    2. "next / last block" in the question, resolved from the student's block and the calendar
       (after Year 1 Block D comes Year 2 Block A; with nothing before or after, the whole year);
    3. the student's own year and block.
    Block names are mapped to the folders the year uses (Year 3: semesters)."""
    named_year, named_blocks = year_block_in_question(question) if config.FILTER_BY_QUESTION_YEAR else ("", [])
    relative = block_ref_in_question(question) if config.FILTER_BY_QUESTION_YEAR else ""
    if named_year or named_blocks:
        year = named_year or user_year
        return (year, segments_for(year, named_blocks)) if year else (None, None)
    if not user_year:
        return None, None
    if relative:
        target = resolve_block(relative, today or date.today(), user_year, user_block or "")
        return (target.year, list(target.blocks)) if target else (user_year, None)
    own = segments_for(user_year, [user_block]) if user_block else None
    return user_year, own


def block_only_target(
    question: str, user_year: str | None, user_block: str | None, today: date | None = None
) -> tuple[str, list[str]] | None:
    """With BLOCK_ONLY_SEARCH: the year and block(s) a question is about, when it names a block ("Year 1 Block C",
    "block c", "Y1C", "semester 2") or says next / last block, and filter_for resolves that to blocks.
    None otherwise (no block in the question, a whole year, nothing before / after): today's search."""
    if not (config.BLOCK_ONLY_SEARCH and config.FILTER_BY_QUESTION_YEAR):
        return None
    if not (year_block_in_question(question)[1] or block_ref_in_question(question)):
        return None
    year, blocks = filter_for(question, user_year, user_block, today)
    return (year, blocks) if year and blocks else None


def search_scope(
    question: str, user_year: str | None, user_block: str | None, today: date | None = None
) -> tuple[str | None, list[str] | None, bool]:
    """(year, blocks, shared): the filter for the searches. shared=False for a question about one block."""
    target = block_only_target(question, user_year, user_block, today)
    if target:
        return target[0], target[1], False
    return *filter_for(question, user_year, user_block, today), True


def retrieve(
    question: str,
    k: int = config.TOP_K,
    user_year: str | None = None,
    user_block: str | None = None,
    today: date | None = None,
) -> list[dict]:
    """user_year / user_block: the student's own year and block (e.g. 'Year2', 'BlockB'), and today's date
    for "next block"; see filter_for. Pages with no year/block stay searchable, except for a question about one
    block when BLOCK_ONLY_SEARCH is on (see search_scope)."""
    year, blocks, shared = search_scope(question, user_year, user_block, today)
    query_vector = embed([question])[0]
    if not config.HYBRID:
        return search(get_client(), query_vector, k, year=year, blocks=blocks, shared=shared)
    depth = max(k, FUSION_DEPTH)
    dense = search(get_client(), query_vector, depth, year=year, blocks=blocks, shared=shared)
    return rrf_merge([dense, bm25_search(question, depth, year, blocks, shared)], k)  # meaning + exact words


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
    queries: list[str],
    k: int,
    question: str,
    user_year: str | None = None,
    user_block: str | None = None,
    today: date | None = None,
) -> list[dict]:
    """Search with several queries and merge with RRF. The filter is decided from the ORIGINAL question,
    so a rewrite can't change which block is searched. With HYBRID, every query also gets a BM25 search."""
    year, blocks, shared = search_scope(question, user_year, user_block, today)
    client = get_client()
    vectors = embed(queries)  # one batch call for all queries
    depth = max(2 * k, FUSION_DEPTH)
    lists = [search(client, v, depth, year=year, blocks=blocks, shared=shared) for v in vectors]
    if config.HYBRID:
        lists += [bm25_search(q, depth, year, blocks, shared) for q in queries]
    return rrf_merge(lists, k)
