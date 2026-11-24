import pydantic
import pytest

from ex05_documents import Document, load_documents, save_documents


def make_docs():
    return [
        Document(id="1", text="First document.", source="a.txt"),
        Document(id="2", text="Second one.", source="b.txt", metadata={"lang": "en"}),
    ]


def test_document_fields_and_default_metadata():
    doc = Document(id="1", text="hello", source="x.txt")
    assert doc.id == "1"
    assert doc.text == "hello"
    assert doc.source == "x.txt"
    assert doc.metadata == {}


def test_metadata_default_is_not_shared():
    first = Document(id="1", text="a", source="s")
    second = Document(id="2", text="b", source="s")
    first.metadata["k"] = "v"
    assert second.metadata == {}


@pytest.mark.parametrize("field", ["id", "text"])
def test_empty_id_or_text_is_invalid(field):
    values = {"id": "1", "text": "hello", "source": "x.txt", field: ""}
    with pytest.raises(pydantic.ValidationError):
        Document(**values)


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "docs.jsonl"
    docs = make_docs()
    save_documents(docs, path)
    assert load_documents(path) == docs


def test_save_writes_one_line_per_document(tmp_path):
    path = tmp_path / "docs.jsonl"
    save_documents(make_docs(), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_save_creates_parent_folders(tmp_path):
    path = tmp_path / "nested" / "deeper" / "docs.jsonl"
    save_documents(make_docs(), path)
    assert path.exists()


def test_load_skips_blank_lines(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"id": "1", "text": "hi", "source": "s"}\n\n   \n{"id": "2", "text": "yo", "source": "s"}\n',
        encoding="utf-8",
    )
    assert [doc.id for doc in load_documents(path)] == ["1", "2"]


def test_load_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"id": "1", "text": "hi", "source": "s"}\n{not json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_documents(path)


def test_load_invalid_document_reports_line_number(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"id": "1", "text": "hi", "source": "s"}\n\n{"id": "2", "source": "s"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 3"):
        load_documents(path)


def test_load_duplicate_ids_raise(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"id": "dup-7", "text": "hi", "source": "s"}\n{"id": "dup-7", "text": "yo", "source": "s"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dup-7"):
        load_documents(path)
