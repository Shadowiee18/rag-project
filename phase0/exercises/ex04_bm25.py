"""Exercise 04: BM25 keyword search.

BM25 is the keyword half of "hybrid search" (Phase 2). Vector search is good at meaning;
BM25 is good at exact words such as names, error codes and product numbers.
Interviewers sometimes ask you to write it from scratch.

Run the tests:  uv run pytest -k ex04
"""


class BM25:
    """Okapi BM25 over a corpus of already-tokenized documents.

    For a query term t and a document d:

        idf(t)      = ln( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
        tf_part     = f(t,d) * (k1 + 1) / ( f(t,d) + k1 * (1 - b + b * |d| / avgdl) )
        score(q, d) = sum, over each term t in the query, of idf(t) * tf_part

    N      = number of documents
    n_t    = number of documents that contain t at least once
    f(t,d) = how many times t appears in d
    |d|    = number of tokens in d
    avgdl  = average number of tokens per document

    A term repeated in the query counts once for each time it appears.
    Use math.log (natural log).
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """Store k1 and b, and precompute what score() needs (N, avgdl, document
        frequencies, per-document term counts) so scoring doesn't re-scan the corpus.

        Raise ValueError if the corpus is empty.
        """
        raise NotImplementedError

    def idf(self, term: str) -> float:
        """Inverse document frequency of `term` (formula above)."""
        raise NotImplementedError

    def score(self, query: list[str], doc_index: int) -> float:
        """BM25 score of document `doc_index` for the tokenized `query`."""
        raise NotImplementedError

    def search(self, query: list[str], k: int) -> list[tuple[int, float]]:
        """Return the top k (doc_index, score) pairs, highest score first.

        When scores are equal, the lower doc_index comes first.
        Leave out documents whose score is 0.
        """
        raise NotImplementedError
