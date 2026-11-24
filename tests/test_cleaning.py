from ragapp.cleaning import canonical, clean, is_template
from ragapp.documents import Document

LONG = "Real lesson text about ARIMA models and stationarity. " * 10


def doc(id_, text, section="Study Content"):
    return Document(id=id_, text=text, source=f"https://x/{id_}", metadata={"section": section, "title": id_})


def test_short_pages_are_dropped():
    report = clean([doc("a", "# Just a heading"), doc("b", LONG)])
    assert report.dropped_short == ["a"]
    assert [d.id for d in report.kept] == ["b"]


def test_exact_duplicates_collapse_to_shortest_path_and_remember_the_copies():
    report = clean([doc("MLOps/MLOps_1/Docs", LONG), doc("MLOps/Docs", LONG), doc("MLOps_1/Docs", "  " + LONG)])
    assert [d.id for d in report.kept] == ["MLOps/Docs"]
    assert report.duplicate_of == {"MLOps/MLOps_1/Docs": "MLOps/Docs", "MLOps_1/Docs": "MLOps/Docs"}
    assert "https://x/MLOps/MLOps_1/Docs" in report.kept[0].metadata["also_at"]


def test_merged_copy_keeps_every_block_it_belongs_to():
    a = Document(id="Year1/BlockA/Att", text=LONG, source="a", metadata={"year": "Year1", "block": "BlockA"})
    b = Document(id="Year1/BlockB/Att", text=LONG, source="b", metadata={"year": "Year1", "block": "BlockB"})
    kept = clean([a, b]).kept[0].metadata
    assert (kept["year"], kept["block"]) == ("Year1", "BlockA BlockB")


def test_merged_copy_with_a_shared_page_stays_shared():
    # a copy with no year/block is visible to every student, so the merged page must be too
    a = Document(id="General/Att", text=LONG, source="a", metadata={"year": "", "block": ""})
    b = Document(id="Year1/BlockD/Att", text=LONG, source="b", metadata={"year": "Year1", "block": "BlockD"})
    kept = clean([a, b]).kept[0].metadata
    assert (kept["year"], kept["block"]) == ("", "")


def test_near_duplicates_are_kept():
    report = clean([doc("S1/PlaySmart", LONG + " 4 iterative phases."), doc("S2/PlaySmart", LONG + " 3 iterative phases.")])
    assert len(report.kept) == 2


def test_template_detection():
    assert is_template(doc("Templates/Pages Template", LONG, section="Templates"))
    assert is_template(doc("x", "This is a template for a page. Write an introduction here. " + LONG))
    assert is_template(doc("brief", "# Project Brief: ShortName\n\n## Introduction\n\n## Scope\n\n" * 5))
    # a real guide that mentions templates once is not a template
    assert not is_template(doc("Staff/4. Pull Request Template", "How to use our pull request template. " + LONG, section="Staff"))


def test_kept_pages_are_tagged_with_page_type():
    report = clean([doc("Templates/T", LONG, section="Templates"), doc("lesson", LONG + " extra")])
    types = {d.id: d.metadata["page_type"] for d in report.kept}
    assert types == {"Templates/T": "template", "lesson": "content"}
    assert report.templates == ["Templates/T"]


def test_canonical_is_stable():
    assert canonical(["b/x", "a/x", "a/b/x"]) == "a/x"
