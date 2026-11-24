import pytest

from ragapp.ingest import chunk_header, label
from ragapp.retrieve import year_block_in_question


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What are the DataLab tasks in Year 2 Block B?", ("Year2", ["BlockB"])),
        ("year2 block a deadline", ("Year2", ["BlockA"])),
        ("Y2B datalab tasks", ("Year2", ["BlockB"])),
        ("I'm a first-year student. How late can I be?", ("Year1", [])),
        ("In second year Block A, which emotions?", ("Year2", ["BlockA"])),
        ("Year 3 semester 1 grades", ("Year3", ["Semester1"])),
        ("What is overfitting?", ("", [])),
        ("Which block covers Docker?", ("", [])),
        ("Block C final submission", ("", ["BlockC"])),
        # the bug the eval found: questions that name two blocks
        ("Which textbooks are used in Year 2 Block A and Year 2 Block B?", ("Year2", ["BlockA", "BlockB"])),
        ("In Year 2, when is the Block C deadline, and when are the Block D deliverables due?", ("Year2", ["BlockC", "BlockD"])),
    ],
)
def test_year_block_in_question(question, expected):
    assert year_block_in_question(question) == expected


def test_label_splits_merged_blocks_into_a_list():
    assert label("BlockA BlockB") == ["BlockA", "BlockB"]
    assert label("BlockB") == "BlockB"
    assert label("") == ""


def test_chunk_header_uses_readable_path_and_title():
    assert chunk_header("Year2/BlockB/DataLab Tasks", "DataLab Tasks") == "Year 2 · Block B · DataLab Tasks"
    assert chunk_header("Year3/Semester1", "Semester1") == "Year 3 · Semester1"
    assert (
        chunk_header("Study Content/Deep Learning/3. Improving Model Performance", "Improving Model Performance")
        == "Study Content · Deep Learning · Improving Model Performance"
    )
