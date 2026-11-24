"""Exercise 04: BM25 keyword search.

BM25 is the keyword half of "hybrid search" (Phase 2). Vector search is good at meaning;
BM25 is good at exact words such as names, error codes and product numbers.
Interviewers sometimes ask you to write it from scratch.

Run the tests:  uv run pytest -k ex04
"""

import math

from ex01_word_freq import word_counts


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
        if not corpus:
            raise ValueError("corpus must contain at least one document")

        self.k1 = k1
        self.b = b

        # N: how many documents there are.
        self.num_docs = len(corpus)

        # |d| for every document, and avgdl (idea 3: length normalisation).
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / self.num_docs

        # f(t, d): word counts for every document, e.g. [{"a": 1, "b": 1}, {"a": 1, "c": 2}, ...]
        # (idea 1: more matches -> higher score).
        self.term_counts = [word_counts(doc) for doc in corpus]

        # n_t: in how many documents each word appears at least once (idea 2: rare words count more).
        # set(doc) removes duplicates, so a word repeated inside one document is counted once.
        self.doc_freq = {}
        for doc in corpus:
            for term in set(doc):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def idf(self, term: str) -> float:
        """Inverse document frequency of `term` (formula above)."""
        n_t = self.doc_freq.get(term, 0)
        return math.log((self.num_docs - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query: list[str], doc_index: int) -> float:
        """BM25 score of document `doc_index` for the tokenized `query`."""
        counts = self.term_counts[doc_index]
        length_ratio = self.doc_lengths[doc_index] / self.avg_doc_length

        total = 0.0
        for term in query:
            f = counts.get(term, 0)
            if f == 0:
                continue  # this word isn't in the document, so it adds nothing

            tf_part = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length_ratio))
            total += self.idf(term) * tf_part
        return total

    def search(self, query: list[str], k: int) -> list[tuple[int, float]]:
        """Return the top k (doc_index, score) pairs, highest score first.

        When scores are equal, the lower doc_index comes first.
        Leave out documents whose score is 0.
        """
        results = []
        for doc_index in range(self.num_docs):
            doc_score = self.score(query, doc_index)
            if doc_score > 0:
                results.append((doc_index, doc_score))

        sorted_results = sorted(results, key=lambda pair: (-pair[1], pair[0]))
        return sorted_results[:k]
