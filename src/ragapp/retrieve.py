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


def retrieve(
    question: str,
    k: int = config.TOP_K,
    user_year: str | None = None,
    user_block: str | None = None,
    today: date | None = None,
) -> list[dict]:
    """user_year / user_block: the student's own year and block (e.g. 'Year2', 'BlockB'), and today's date
    for "next block"; see filter_for. Pages with no year/block always stay searchable."""
    year, blocks = filter_for(question, user_year, user_block, today)
    query_vector = embed([question])[0]
    if not config.HYBRID and config.SCOPE_MODE != "boost":
        return search(get_client(), query_vector, k, year=year, blocks=blocks)
    depth = max(k, FUSION_DEPTH)
    return rrf_merge(search_lists(get_client(), [query_vector], [question], depth, year, blocks), k)


def search_lists(client, vectors: list, queries: list[str], depth: int, year: str | None, blocks: list[str] | None) -> list:
    """The result lists to merge: a vector search per query (+ BM25 per query with HYBRID), for each scope.

    SCOPE_MODE "filter": one scope, the year/block plus shared pages.
    SCOPE_MODE "boost": everything (no year/block filter), and the scope's own pages (no shared ones), so the
    scope's pages are candidates even in a large corpus; scope_boost then decides the order.
    """
    if config.SCOPE_MODE == "boost":
        scopes = [(None, None, True)] + ([(year, blocks, False)] if year else [])
    else:
        scopes = [(year, blocks, True)]
    lists = []
    for scope_year, scope_blocks, shared in scopes:
        lists += [search(client, v, depth, year=scope_year, blocks=scope_blocks, shared=shared) for v in vectors]
        if config.HYBRID:
            lists += [bm25_search(q, depth, scope_year, scope_blocks, shared) for q in queries]
    return lists


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def scope_boost(
    chunks: list[dict], year: str | None, blocks: list[str] | None, w_block: float, w_year: float
) -> list[dict]:
    """Re-sort chunks by their score scaled to 0-1 within this question (min-max), plus w_year for a page of the
    scope's year and w_block for a page of its year AND one of its blocks. Shared pages get no bonus and no penalty.

    Scaling first makes one weight fit every question: ColBERT scores grow with the question's length.
    """
    if not year or not chunks:
        return chunks
    scores = [c["score"] for c in chunks]
    low, span = min(scores), (max(scores) - min(scores)) or 1.0

    def bonus(chunk: dict) -> float:
        in_year = year in _as_list(chunk.get("year", ""))
        in_block = in_year and bool(blocks) and any(b in _as_list(chunk.get("block", "")) for b in blocks)
        return w_year * in_year + w_block * in_block

    boosted = [{**c, "score": (c["score"] - low) / span + bonus(c)} for c in chunks]
    return sorted(boosted, key=lambda c: -c["score"])


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
    year, blocks = filter_for(question, user_year, user_block, today)
    vectors = embed(queries)  # one batch call for all queries
    depth = max(2 * k, FUSION_DEPTH)
    return rrf_merge(search_lists(get_client(), vectors, queries, depth, year, blocks), k)
