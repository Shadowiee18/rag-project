import math

import pytest

from ex03_similarity import cosine_similarity, dot, norm, top_k


def test_dot():
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert dot([], []) == 0


def test_dot_length_mismatch_raises():
    with pytest.raises(ValueError):
        dot([1, 2], [1, 2, 3])


def test_norm():
    assert norm([3, 4]) == pytest.approx(5.0)
    assert norm([0, 0, 0]) == 0


def test_cosine_same_direction_is_one():
    assert cosine_similarity([1, 2], [2, 4]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine_similarity([1, 0], [0, 5]) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one():
    assert cosine_similarity([1, 1], [-2, -2]) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0
    assert cosine_similarity([1, 2], [0, 0]) == 0.0


def test_cosine_length_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_top_k_ranks_by_similarity():
    docs = {"a": [1, 0], "b": [0, 1], "c": [1, 1], "d": [-1, 0]}
    result = top_k([1, 0], docs, k=2)
    assert [doc_id for doc_id, _ in result] == ["a", "c"]
    assert result[0][1] == pytest.approx(1.0)
    assert result[1][1] == pytest.approx(1 / math.sqrt(2))


def test_top_k_ties_are_alphabetical():
    docs = {"y": [1, 0], "x": [2, 0]}
    assert [doc_id for doc_id, _ in top_k([1, 0], docs, k=2)] == ["x", "y"]


def test_top_k_more_than_available():
    docs = {"a": [1, 0], "b": [0, 1]}
    assert len(top_k([1, 0], docs, k=10)) == 2
