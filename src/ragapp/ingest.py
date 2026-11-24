"""Indexing time: load pages -> chunk -> embed -> store in Qdrant.

Run:  uv run python -m ragapp.ingest
"""

import re
import time

from ragapp import config
from ragapp.chunking import chunk_text
from ragapp.cleaning import clean
from ragapp.documents import load_documents
from ragapp.embed import embed
from ragapp.store import add_chunks, get_client, recreate_collection

BATCH_SIZE = 32


def readable(folder: str) -> str:
    """'Year2' -> 'Year 2', 'BlockB' -> 'Block B', 'Semester1' -> 'Semester 1'."""
    return re.sub(r"^(Year|Block|Semester)(\w)$", r"\1 \2", folder)


def chunk_header(doc_id: str, title: str) -> str:
    """Breadcrumb from the page's path: 'Year2/BlockB/DataLab Tasks' -> 'Year 2 · Block B · DataLab Tasks'."""
    folders = [readable(part) for part in doc_id.split("/")[:-1]]
    return " · ".join(folders + [title or doc_id])


def label(value: str) -> str | list[str]:
    """'BlockA BlockB' (a merged duplicate, see cleaning.py) -> ['BlockA', 'BlockB'].
    Qdrant's MatchAny filter matches a list field if any of its values matches."""
    return value.split() if " " in value else value


def load_corpus():
    docs = load_documents(config.CORPUS_PATH)
    if not config.CLEAN_CORPUS:
        return docs
    report = clean(docs)
    print(f"cleaning: kept {len(report.kept)} of {len(docs)} pages "
          f"({len(report.dropped_short)} near-empty, {len(report.duplicate_of)} duplicate copies merged, "
          f"{len(report.templates)} tagged template)")
    return report.kept


def build_chunks() -> list[dict]:
    """Split every document into chunks. Each chunk keeps where it came from."""
    chunks = []
    for doc in load_corpus():
        title = doc.metadata.get("title", "")
        header = chunk_header(doc.id, title)
        for index, text in enumerate(chunk_text(doc.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            if config.CHUNK_HEADERS:
                text = f"{header}\n\n{text}"
            chunks.append({
                "doc_id": doc.id,
                "title": title,
                "source": doc.source,
                "section": doc.metadata.get("section", ""),
                "year": label(doc.metadata.get("year", "")),
                "block": label(doc.metadata.get("block", "")),
                "page_type": doc.metadata.get("page_type", ""),
                "chunk_index": index,
                "text": text,
            })
    return chunks


def main() -> None:
    started = time.perf_counter()
    chunks = build_chunks()
    print(f"{len(chunks)} chunks from {config.CORPUS_PATH.name}")

    client = get_client()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = embed([chunk["text"] for chunk in batch])
        if start == 0:
            recreate_collection(client, vector_size=len(vectors[0]))
            print(f"vector size: {len(vectors[0])}")
        add_chunks(client, start_id=start, vectors=vectors, payloads=batch)
        print(f"  embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}", end="\r")

    print(f"\ndone in {time.perf_counter() - started:.0f}s -> collection '{config.COLLECTION}'")


if __name__ == "__main__":
    main()
