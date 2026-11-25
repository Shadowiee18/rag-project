"""The eval's own scoring rules: judge label parsing and refusal detection (no Ollama needed)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import run_eval  # noqa: E402


def fake_judge_reply(monkeypatch, text: str) -> None:
    reply = SimpleNamespace(message=SimpleNamespace(content=text))
    monkeypatch.setattr(run_eval.ollama, "chat", lambda **kwargs: reply)


@pytest.mark.parametrize(
    ("reply", "label"),
    [
        ("CORRECT", "CORRECT"),
        ("correct.", "CORRECT"),
        ("PARTIAL", "PARTIAL"),
        ("Partially correct", "PARTIAL"),
        ("WRONG", "WRONG"),
        ("INCORRECT", "WRONG"),  # contains "CORRECT", but means wrong
        ("Incorrect.", "WRONG"),
        ("I am not sure", "WRONG"),  # no label at all
    ],
)
def test_judge_reads_the_label_as_a_whole_word(monkeypatch, reply, label):
    fake_judge_reply(monkeypatch, reply)
    assert run_eval.judge("q", "expected", "answer") == label


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know",
        "I don’t know.",  # curly apostrophe, as LLMs often write it
        "I do not know",
        "The context does not say when the deadline is. I don't know [2].",
    ],
)
def test_refusal_is_detected_with_any_apostrophe(answer):
    assert run_eval.is_refusal(answer)


def test_an_answer_is_not_a_refusal():
    assert not run_eval.is_refusal("The deadline is Friday of week 5 [1].")


@pytest.mark.parametrize("answer", ["I don't know", "I don’t know.", " I do not know [1][3]. "])
def test_only_refusal_is_an_answer_with_nothing_else_in_it(answer):
    assert run_eval.only_refusal(answer)


def test_rejudge_regrades_a_saved_run_in_place(monkeypatch, tmp_path):
    import rejudge

    rows = [
        {"question": "q1", "expected_answer": "e", "answer": "It is Friday [1].", "source_ids": ["p"],
         "judge": "CORRECT", "refused": False, "cites": True},  # was misread: the judge said INCORRECT
        {"question": "q2", "expected_answer": "e", "answer": "X [1]. For Y, I don’t know.", "source_ids": ["p"],
         "judge": "WRONG", "refused": False, "cites": True},  # half answer, was never judged
        {"question": "q3", "expected_answer": "", "answer": "I don’t know", "source_ids": [],
         "refused": False, "cites": False},  # unanswerable, curly refusal was missed
    ]
    (tmp_path / "run.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (tmp_path / "run_summary.json").write_text(json.dumps({"retrieval": {"hit@1": 0.5}, "config": {}}), encoding="utf-8")
    monkeypatch.setattr(rejudge, "grade", lambda r: {"q1": "WRONG", "q2": "PARTIAL"}[r["question"]])

    assert rejudge.outdated_runs(tmp_path) == ["run"]
    assert rejudge.rejudge("run", tmp_path) == 2
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["answers (answerable, LLM-judged)"]["partial"] == 0.5
    assert summary["unanswerable_correctly_refused"] == 1.0
    assert summary["retrieval"] == {"hit@1": 0.5}  # untouched
    assert rejudge.outdated_runs(tmp_path) == []  # graded under the current rules now


def test_a_partial_answer_is_not_only_a_refusal():
    """Half an answer ("Block B uses X [1]. For Block A I don't know.") must reach the judge, which can say PARTIAL."""
    assert not run_eval.only_refusal("Block B uses the course reader [1]. For Block A, I don't know [3].")
