"""Grouping eval questions by question type and by segment (where the answer lives)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import groups  # noqa: E402

PAGES = {
    "y2b/deadlines": ("Year2", "BlockB"),
    "y2b/tasks": ("Year2", "BlockB"),
    "y2a/tasks": ("Year2", "BlockA"),
    "y3": ("Year3", ""),
    "study/vpn": ("", ""),
}


def test_segment_is_where_the_answer_lives():
    assert groups.segment(["y2b/deadlines"], PAGES) == "Year2 BlockB"
    assert groups.segment(["y2b/deadlines", "y2b/tasks"], PAGES) == "Year2 BlockB"  # same block twice
    assert groups.segment(["y3"], PAGES) == "Year3"  # a year page without a block
    assert groups.segment(["study/vpn"], PAGES) == "shared"  # pages of no year
    assert groups.segment(["y2b/tasks", "y2a/tasks"], PAGES) == "several"
    assert groups.segment([], PAGES) == "none"  # unanswerable


def result(qtype, segment, hit_rank=None, judge=None, evidence=None, refused=False, answerable=True):
    r = {"question_type": qtype, "segment": segment, "source_ids": ["p"] if answerable else [], "refused": refused}
    if answerable:
        r.update(hit_rank=hit_rank, judge=judge)
    if evidence is not None:
        r["evidence_found@5"] = evidence
    return r


def test_group_summary_counts_per_group():
    results = [
        result("rules", "shared", hit_rank=1, judge="CORRECT", evidence=1.0),
        result("rules", "shared", hit_rank=7, judge="WRONG", evidence=0.5),
        result("deadline", "Year2 BlockB", hit_rank=3, judge="PARTIAL"),
        result("out_of_scope", "none", refused=True, answerable=False),
    ]
    by_type = groups.group_summary(results, "question_type")
    assert by_type["rules"] == {"questions": 2, "hit@5": 0.5, "all_evidence@5": 0.5, "correct": 0.5}
    assert by_type["deadline"] == {"questions": 1, "hit@5": 1.0, "correct": 0.0}  # no evidence phrases: no number
    assert by_type["out_of_scope"] == {"questions": 1, "refused": 1.0}


def test_coverage_shows_empty_cells_for_every_type_and_site_segment():
    items = [
        {"question_type": "rules", "segment": "shared", "split": "dev"},
        {"question_type": "rules", "segment": "shared", "split": "test"},
        {"question_type": "deadline", "segment": "Year2 BlockB", "split": "dev"},
    ]
    table = groups.coverage(items, PAGES)
    assert table["rules"]["shared"] == {"dev": 1, "test": 1}
    assert table["deadline"]["Year2 BlockB"] == {"dev": 1, "test": 0}
    assert table["people"]["Year2 BlockA"] == {"dev": 0, "test": 0}  # a type and a block nobody asked about
    assert list(table["rules"])[:4] == ["Year2 BlockA", "Year2 BlockB", "Year3", "shared"]  # site order, then the rest
