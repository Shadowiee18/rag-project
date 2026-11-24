"""Exercise 05: a Document model and JSONL files.

JSONL = one JSON object per line. It's the usual format for datasets and eval sets
because you can stream it line by line and append to it.

Useful: Document.model_validate_json(line), doc.model_dump_json(),
pydantic.Field(min_length=1), pathlib.Path.

Run the tests:  uv run pytest -k ex05
"""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class Document(BaseModel):
    """One document. Replace `pass` with these fields:

    - id: str, at least 1 character
    - text: str, at least 1 character
    - source: str (where it came from, such as a filename or URL)
    - metadata: dict[str, str], defaulting to an empty dict
    """

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)


def load_documents(path: Path) -> list[Document]:
    """Read a JSONL file into a list of Documents, in file order.

    Skip blank lines.
    If a line is not valid JSON or not a valid Document, raise ValueError with a message
    that contains "line N", where N is the 1-based line number in the file.
    If two documents have the same id, raise ValueError with a message that contains that id.
    """
    docs = []
    seen_ids = set()

    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue  # blank line

        try:
            doc = Document.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(f"{path}, line {line_number}: {error}") from error

        if doc.id in seen_ids:
            raise ValueError(f"{path}, line {line_number}: duplicate id {doc.id!r}")
        seen_ids.add(doc.id)
        docs.append(doc)

    return docs


def save_documents(docs: list[Document], path: Path) -> None:
    """Write `docs` to `path` as JSONL, one document per line, in order.

    Create the parent folders if they don't exist.
    load_documents(path) must give back exactly the same documents.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [doc.model_dump_json() + "\n" for doc in docs]
    path.write_text("".join(lines), encoding="utf-8")
