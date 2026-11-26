"""The scope tuning eval (evals/scope_eval.py): counter-examples, scoring and the weight rule. No models needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import scope_eval as se  # noqa: E402

META = {"Study Content/VPN": ("", ""), "General/Rules": ("", ""), "Year2/BlockC": ("Year2", "BlockC")}


def item(id_, sources, question="how do I connect to the VPN?"):
    return {"id": id_, "question": question, "source_ids": sources}


def test_readable():
    assert se.readable("Year1", "BlockC") == "Year 1 Block C"
    assert se.readable("Year3", "Semester2") == "Year 3 Semester 2"


def test_counter_cases_are_only_questions_answered_by_shared_pages():
    items = [item("a", ["Study Content/VPN"]), item("b", ["Year2/BlockC"]), item("c", []),
             item("d", ["General/Rules", "Year2/BlockC"]), item("e", ["General/Rules"])]
    assert [c["id"] for c in se.counter_cases(items, META)] == ["counter-a", "counter-e"]


def test_counter_cases_put_a_block_in_front_rotating_over_the_segments():
    items = [item(str(n), ["Study Content/VPN"]) for n in range(11)]
    cases = se.counter_cases(items, META)
    assert cases[0]["question"] == "Year 1 Block A: how do I connect to the VPN?"
    assert cases[9]["question"].startswith("Year 3 Semester 2: ")
    assert cases[10]["question"].startswith("Year 1 Block A: ")
    assert all(c["set"] == "counter" and c["source_ids"] == ["Study Content/VPN"] for c in cases)


def chunk(doc_id, year="", block=""):
    return {"doc_id": doc_id, "year": year, "block": block, "score": 1.0}


def test_hits_for_a_page_question():
    case = {"set": "page", "sources": ["p"]}
    assert se.hits(case, [chunk("p")]) == (True, True)
    assert se.hits(case, [chunk(str(n)) for n in range(4)] + [chunk("p")]) == (False, True)
    assert se.hits(case, [chunk(str(n)) for n in range(5)] + [chunk("p")]) == (False, False)


def test_hits_for_a_directional_case_read_the_segment():
    case = {"set": "dir", "target": ["Year2", "BlockC"]}
    assert se.hits(case, [chunk("x", "Year2", "BlockC")]) == (True, True)
    assert se.hits(case, [chunk("x"), chunk("y", ["Year2"], ["BlockB", "BlockC"])]) == (False, True)
    assert se.hits(case, [chunk("x", "Year1", "BlockC")]) == (False, False)
    assert se.hits(case, []) == (False, False)


def scores(page5, student5, counter5, extra=0):
    row = lambda h5: {"n": 10, "hit@1": 0, "hit@5": h5}
    return {"page": row(page5), "student": row(student5), "counter": row(counter5), "dir": {"n": 18, "hit@1": extra, "hit@5": extra}}


def test_choose_drops_weights_that_lose_hits_at_5_anywhere():
    baseline = scores(9, 6, 8)
    grid = {(0.0, 0.0): scores(9, 6, 8), (0.5, 0.0): scores(9, 6, 7, extra=18), (0.2, 0.0): scores(9, 6, 8, extra=10)}
    assert se.choose(grid, baseline) == (0.2, 0.0)  # 0.5 has more hits but loses a counter-example


def test_choose_prefers_smaller_weights_on_a_tie():
    baseline = scores(9, 6, 8)
    grid = {(0.3, 0.1): scores(9, 6, 8, extra=5), (0.1, 0.0): scores(9, 6, 8, extra=5)}
    assert se.choose(grid, baseline) == (0.1, 0.0)


def test_choose_gives_none_when_nothing_passes():
    assert se.choose({(0.1, 0.0): scores(8, 6, 8)}, scores(9, 6, 8)) is None


def test_tune_scores_filter_and_boost_candidates(monkeypatch):
    shared, block = chunk("shared"), chunk("y2c", "Year2", "BlockC")
    shared["score"], block["score"] = 20.0, 19.0
    low = {**chunk("other"), "score": 10.0}
    cases = [{"set": "dir", "target": ["Year2", "BlockC"], "scope": ["Year2", ["BlockC"]],
              "filter": [shared, block, low], "boost": [shared, block, low]},
             {"set": "counter", "sources": ["shared"], "scope": ["Year1", ["BlockA"]],
              "filter": [shared, low], "boost": [shared, low]}]
    out = se.tune(cases)
    assert out["filter"]["dir"]["hit@1"] == 0
    assert out["chosen"]["scores"]["dir"]["hit@1"] == 1 and out["chosen"]["scores"]["counter"]["hit@5"] == 1
