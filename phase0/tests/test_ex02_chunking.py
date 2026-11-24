import pytest

from ex02_chunking import chunk_text


def test_docstring_examples():
    assert chunk_text("abcdefghij", chunk_size=4, overlap=1) == ["abcd", "defg", "ghij"]
    assert chunk_text("abcdefghijk", chunk_size=4, overlap=1) == ["abcd", "defg", "ghij", "jk"]


def test_no_overlap():
    assert chunk_text("abcdef", chunk_size=3, overlap=0) == ["abc", "def"]


def test_large_overlap():
    assert chunk_text("abcdefgh", chunk_size=4, overlap=2) == ["abcd", "cdef", "efgh"]


def test_text_shorter_than_or_equal_to_chunk_size():
    assert chunk_text("abc", chunk_size=10, overlap=2) == ["abc"]
    assert chunk_text("abcd", chunk_size=4, overlap=1) == ["abcd"]


def test_empty_text():
    assert chunk_text("", chunk_size=4, overlap=1) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)],
)
def test_invalid_arguments_raise(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)


@pytest.mark.parametrize(
    ("text", "chunk_size", "overlap"),
    [
        ("The quick brown fox jumps over the lazy dog.", 10, 3),
        ("x" * 101, 20, 5),
        ("short", 2, 1),
        ("a b c d e f g h i j k l m n o p", 7, 0),
    ],
)
def test_chunks_rebuild_the_original_text(text, chunk_size, overlap):
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    rebuilt = chunks[0] + "".join(chunk[overlap:] for chunk in chunks[1:])
    assert rebuilt == text
    assert all(len(chunk) == chunk_size for chunk in chunks[:-1])
    assert 0 < len(chunks[-1]) <= chunk_size
