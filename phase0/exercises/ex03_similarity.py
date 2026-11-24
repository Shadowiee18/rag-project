"""Exercise 03: vector similarity in plain Python (no numpy).

This is what a vector database does at its core: find the stored vectors
pointing in the most similar direction to the query vector.

Run the tests:  uv run pytest -k ex03
"""

import math


def dot(a: list[float], b: list[float]) -> float:
    """Dot product: sum of a[i] * b[i]. Raise ValueError if the lengths differ.

    >>> dot([1, 2, 3], [4, 5, 6])
    32
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")

    total = 0
    for x, y in zip(a, b):
        total += x * y
    return total


def norm(a: list[float]) -> float:
    """Euclidean length: square root of the sum of squares.

    >>> norm([3, 4])
    5.0
    """
    return math.sqrt(dot(a, a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """dot(a, b) / (norm(a) * norm(b)).

    Ranges from -1 (opposite) through 0 (unrelated) to 1 (same direction).
    Raise ValueError if the lengths differ.
    Return 0.0 if either vector is all zeros (you can't divide by zero).
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same length")

    length_a = norm(a)
    length_b = norm(b)
    if length_a == 0 or length_b == 0:
        return 0.0

    return dot(a, b) / (length_a * length_b)


def top_k(query: list[float], docs: dict[str, list[float]], k: int) -> list[tuple[str, float]]:
    """Return the k doc ids most similar to `query` as (doc_id, similarity) pairs.

    Use cosine similarity. Sort highest first. When scores are equal, sort ids alphabetically.
    If there are fewer than k docs, return all of them.
    """
    scores = []
    for doc_id, vector in docs.items():
        scores.append((doc_id, cosine_similarity(query, vector)))

    sorted_scores = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
    return sorted_scores[:k]
