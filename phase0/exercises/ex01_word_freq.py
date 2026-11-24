"""Exercise 01: word frequency.

Run the tests:  uv run pytest -k ex01
"""


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens.

    A token is a run of letters a-z or digits 0-9, after lowercasing.
    Everything else (spaces, punctuation, newlines) separates tokens.

    >>> tokenize("Hello, World! RAG-2026")
    ['hello', 'world', 'rag', '2026']

    Hint: the `re` module.
    """
    raise NotImplementedError


def word_counts(tokens: list[str]) -> dict[str, int]:
    """Count how many times each token appears.

    >>> word_counts(["a", "b", "a"])
    {'a': 2, 'b': 1}

    Write it with a plain dict first. Then look up collections.Counter.
    """
    raise NotImplementedError


def top_k_words(text: str, k: int) -> list[tuple[str, int]]:
    """Return the k most frequent tokens in `text` as (token, count) pairs.

    Sort by count, highest first. When counts are equal, sort the tokens alphabetically.
    If there are fewer than k distinct tokens, return all of them.

    >>> top_k_words("b a b c a b", 2)
    [('b', 3), ('a', 2)]

    Hint: sorted(..., key=...) with a tuple key.
    """
    raise NotImplementedError
