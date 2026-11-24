"""Exercise 03: vector similarity in plain Python (no numpy).

This is what a vector database does at its core: find the stored vectors
pointing in the most similar direction to the query vector.

Run the tests:  uv run pytest -k ex03
"""


def dot(a: list[float], b: list[float]) -> float:
    """Dot product: sum of a[i] * b[i]. Raise ValueError if the lengths differ.

    >>> dot([1, 2, 3], [4, 5, 6])
    32
    """
    raise NotImplementedError


def norm(a: list[float]) -> float:
    """Euclidean length: square root of the sum of squares.

    >>> norm([3, 4])
    5.0
    """
    raise NotImplementedError


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """dot(a, b) / (norm(a) * norm(b)).

    Ranges from -1 (opposite) through 0 (unrelated) to 1 (same direction).
    Raise ValueError if the lengths differ.
    Return 0.0 if either vector is all zeros (you can't divide by zero).
    """
    raise NotImplementedError


def top_k(query: list[float], docs: dict[str, list[float]], k: int) -> list[tuple[str, float]]:
    """Return the k doc ids most similar to `query` as (doc_id, similarity) pairs.

    Use cosine similarity. Sort highest first. When scores are equal, sort ids alphabetically.
    If there are fewer than k docs, return all of them.
    """
    raise NotImplementedError
