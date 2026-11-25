"""The eval's own scoring rules: judge label parsing and refusal detection (no Ollama needed)."""

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


def test_a_partial_answer_is_not_only_a_refusal():
    """Half an answer ("Block B uses X [1]. For Block A I don't know.") must reach the judge, which can say PARTIAL."""
    assert not run_eval.only_refusal("Block B uses the course reader [1]. For Block A, I don't know [3].")
