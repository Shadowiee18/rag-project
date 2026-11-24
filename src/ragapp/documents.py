"""Document model and JSONL loading (from phase0 ex05)."""

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class Document(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str
    metadata: dict[str, str] = Field(default_factory=dict)


def load_documents(path: Path) -> list[Document]:
    docs = []
    seen_ids = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            doc = Document.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(f"{path}, line {line_number}: {error}") from error
        if doc.id in seen_ids:
            raise ValueError(f"{path}, line {line_number}: duplicate id {doc.id!r}")
        seen_ids.add(doc.id)
        docs.append(doc)
    return docs
