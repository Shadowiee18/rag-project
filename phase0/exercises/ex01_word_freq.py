"""Exercise 01: word frequency.

Run the tests:  uv run pytest -k ex01
"""

def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens.

    A token is a run of letters a-z or digits 0-9, after lowercasing.
    Everything else (spaces, punctuation, newlines) separates tokens.

    >>> tokenize("Hello, World! RAG-2026")
    ['hello', 'world', 'rag', '2026']

    Hint: no regex needed. Plain string methods are enough, but text.lower().split()
    on its own won't pass: "Hello," would keep its comma. Deal with the punctuation
    first, character by character, then split. str.isalnum() can help.
    """


    tokens = "".join(char if char.isalnum() else " " for char in text).lower().split()
    return tokens



def word_counts(tokens: list[str]) -> dict[str, int]:
    """Count how many times each token appears.

    >>> word_counts(["a", "b", "a"])
    {'a': 2, 'b': 1}

    Write it with a plain dict first. Then look up collections.Counter.
    """
    word_counter = {}
    for token in tokens:
        if token not in word_counter:
            word_counter[token] = 1
        else:
            word_counter[token] += 1
    return word_counter


def top_k_words(text: str, k: int) -> list[tuple[str, int]]:
    """Return the k most frequent tokens in `text` as (token, count) pairs.

    Sort by count, highest first. When counts are equal, sort the tokens alphabetically.
    If there are fewer than k distinct tokens, return all of them.

    >>> top_k_words("b a b c a b", 2)
    [('b', 3), ('a', 2)]

    Hint: sorted(..., key=...) with a tuple key.
    """
    tokens = tokenize(text)
    counts = word_counts(tokens)
    sorted_pairs = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return sorted_pairs[:k]
