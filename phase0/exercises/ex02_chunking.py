"""Exercise 02: chunking text.

RAG systems split documents into overlapping chunks before embedding them.
Overlap means a sentence cut at a chunk boundary still appears whole in one of the chunks.

Run the tests:  uv run pytest -k ex02
"""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into chunks of `chunk_size` characters.

    Each chunk starts (chunk_size - overlap) characters after the previous one,
    so neighbouring chunks share `overlap` characters.
    Stop as soon as a chunk reaches the end of the text.
    The last chunk may be shorter than chunk_size.

    Empty text returns [].
    Raise ValueError if chunk_size < 1, overlap < 0, or overlap >= chunk_size.

    >>> chunk_text("abcdefghij", chunk_size=4, overlap=1)
    ['abcd', 'defg', 'ghij']
    >>> chunk_text("abcdefghijk", chunk_size=4, overlap=1)
    ['abcd', 'defg', 'ghij', 'jk']
    """
    raise NotImplementedError
