"""Fixed-size character chunking with overlap (from phase0 ex02).

This is the naive baseline on purpose: it cuts mid-sentence and ignores headings.
Phase 2 will compare smarter chunking against it.
"""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be at least 1 and overlap must be between 0 and chunk_size - 1")
    if not text:
        return []

    chunks = []
    stop = max(len(text) - overlap, 1)
    for i in range(0, stop, chunk_size - overlap):
        chunks.append(text[i : i + chunk_size])
    return chunks
