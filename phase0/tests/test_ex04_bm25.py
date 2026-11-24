import math

import pytest

from ex04_bm25 import BM25

CORPUS = [
    ["a", "b"],
    ["a", "c", "c"],
    ["d"],
]


def test_empty_corpus_raises():
    with pytest.raises(ValueError):
        BM25([])


def test_rare_terms_have_higher_idf():
    bm25 = BM25(CORPUS)
    assert bm25.idf("c") > bm25.idf("a")


def test_idf_stays_positive_for_a_term_in_every_document():
    bm25 = BM25([["x"], ["x", "y"]])
    assert bm25.idf("x") > 0


def test_exact_score_matches_formula():
    bm25 = BM25(CORPUS)
    n_docs, n_t, f, doc_len, avgdl, k1, b = 3, 1, 2, 3, 2.0, 1.5, 0.75
    idf = math.log((n_docs - n_t + 0.5) / (n_t + 0.5) + 1)
    tf_part = f * (k1 + 1) / (f + k1 * (1 - b + b * doc_len / avgdl))
    assert bm25.score(["c"], 1) == pytest.approx(idf * tf_part)


def test_repeated_query_terms_count_each_time():
    bm25 = BM25(CORPUS)
    assert bm25.score(["c", "c"], 1) == pytest.approx(2 * bm25.score(["c"], 1))


def test_document_without_query_terms_scores_zero():
    bm25 = BM25(CORPUS)
    assert bm25.score(["c"], 0) == 0
    assert bm25.score(["unknown"], 1) == 0


def test_search_prefers_shorter_document_for_same_term_count():
    bm25 = BM25(CORPUS)
    result = bm25.search(["a"], k=5)
    assert [index for index, _ in result] == [0, 1]
    assert result[0][1] > result[1][1]


def test_search_respects_k():
    bm25 = BM25(CORPUS)
    assert len(bm25.search(["a"], k=1)) == 1


def test_search_with_no_matches_is_empty():
    bm25 = BM25(CORPUS)
    assert bm25.search(["unknown"], k=3) == []


def test_b_zero_ignores_document_length_and_ties_keep_index_order():
    bm25 = BM25(CORPUS, b=0.0)
    result = bm25.search(["a"], k=5)
    assert [index for index, _ in result] == [0, 1]
    assert result[0][1] == pytest.approx(result[1][1])
