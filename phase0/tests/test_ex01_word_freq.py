from ex01_word_freq import tokenize, top_k_words, word_counts


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Hello, World! RAG-2026") == ["hello", "world", "rag", "2026"]


def test_tokenize_handles_whitespace_and_newlines():
    assert tokenize("  one\ttwo\nthree  ") == ["one", "two", "three"]


def test_tokenize_empty_and_punctuation_only():
    assert tokenize("") == []
    assert tokenize("!!! ... ???") == []


def test_word_counts():
    assert word_counts(["a", "b", "a"]) == {"a": 2, "b": 1}
    assert word_counts([]) == {}


def test_top_k_words_sorted_by_count():
    assert top_k_words("b a b c a b", 2) == [("b", 3), ("a", 2)]


def test_top_k_words_ties_are_alphabetical():
    assert top_k_words("z y x", 2) == [("x", 1), ("y", 1)]


def test_top_k_words_k_larger_than_vocabulary():
    assert top_k_words("Cat cat dog", 10) == [("cat", 2), ("dog", 1)]


def test_top_k_words_k_zero():
    assert top_k_words("a b c", 0) == []
