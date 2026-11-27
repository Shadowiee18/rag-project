"""evals/push_scores.py: eval grades become Langfuse scores on the question's trace. No network needed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
import push_scores as ps  # noqa: E402

TRACE = "0123456789abcdef0123456789abcdef"
RESULTS = [
    {"id": "a01", "trace_id": TRACE, "source_ids": ["Year2/BlockC"], "hit_rank": 3},
    {"id": "a02", "trace_id": "f" * 32, "source_ids": ["General/VPN"], "hit_rank": None},
    {"id": "u01", "trace_id": "e" * 32, "source_ids": [], "hit_rank": None},  # unanswerable
    {"id": "a03", "trace_id": None, "source_ids": ["x"], "hit_rank": 1},  # run without tracing
]
GRADES = {"a01": {"id": "a01", "correctness": "partial", "faithfulness": "faithful", "citations": "ok",
                  "reason": "misses the second deadline"}}


def by(scores, trace, name):
    (score,) = [s for s in scores if s["traceId"] == trace and s["name"] == name]
    return score


def test_judge_grades_are_categorical_scores_on_the_trace():
    scores = ps.build_scores("run1", RESULTS, GRADES)
    assert by(scores, TRACE, "correctness")["value"] == "partial"
    assert by(scores, TRACE, "correctness")["dataType"] == "CATEGORICAL"
    assert by(scores, TRACE, "faithfulness")["value"] == "faithful"
    assert by(scores, TRACE, "citations")["value"] == "ok"
    assert by(scores, TRACE, "correctness")["metadata"] == {"run": "run1"}


def test_retrieval_scores_for_answerable_questions():
    scores = ps.build_scores("run1", RESULTS, GRADES)
    assert by(scores, TRACE, "right_page_rank") == {**by(scores, TRACE, "right_page_rank"), "value": 3.0, "dataType": "NUMERIC"}
    assert by(scores, TRACE, "right_page_in_context")["value"] == 1.0  # rank 3 <= 5
    assert by(scores, "f" * 32, "right_page_in_context")["value"] == 0.0  # not found
    assert not [s for s in scores if s["traceId"] == "f" * 32 and s["name"] == "right_page_rank"]  # no rank to give


def test_unanswerable_and_untraced_questions():
    scores = ps.build_scores("run1", RESULTS, GRADES)
    assert not [s for s in scores if s["traceId"] == "e" * 32]  # no retrieval score without a right page
    assert all(s["traceId"] for s in scores)  # a03 has no trace: skipped


def test_reason_only_as_comment_with_trace_content():
    assert "comment" not in by(ps.build_scores("run1", RESULTS, GRADES), TRACE, "correctness")
    assert by(ps.build_scores("run1", RESULTS, GRADES, with_comments=True), TRACE, "correctness")["comment"] == \
        "misses the second deadline"


def test_score_ids_are_stable_per_run_trace_and_name():
    first, again = ps.build_scores("run1", RESULTS, GRADES), ps.build_scores("run1", RESULTS, GRADES)
    assert [s["id"] for s in first] == [s["id"] for s in again]  # sending again replaces, no duplicates
    assert len({s["id"] for s in first}) == len(first)
    assert by(ps.build_scores("run2", RESULTS, GRADES), TRACE, "correctness")["id"] != by(first, TRACE, "correctness")["id"]


class _Reply:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.text = status, body or {}, str(body)

    def json(self):
        return self._body


def test_send_posts_lists_in_batches(monkeypatch):
    import httpx

    sent = []
    monkeypatch.setattr(httpx, "post", lambda url, **kw: sent.append((url, kw)) or _Reply(202))
    scores = [{"id": str(n), "name": "x", "value": 1.0} for n in range(250)]
    assert ps.send(scores, "http://localhost:3000/", "pk-lf-1", "sk-lf-1") == (250, [])
    assert [len(kw["json"]) for _, kw in sent] == [100, 100, 50]
    assert sent[0][0] == "http://localhost:3000/api/public/scores"
    assert sent[0][1]["auth"] == ("pk-lf-1", "sk-lf-1")


def test_send_reports_a_partly_rejected_batch_without_retrying(monkeypatch):
    import httpx

    calls = []
    monkeypatch.setattr(httpx, "post", lambda url, **kw: calls.append(1) or _Reply(207, {"accepted": 1, "errors": ["bad"]}))
    accepted, errors = ps.send([{"id": "1"}, {"id": "2"}], "http://h", "pk", "sk")
    assert (accepted, len(calls), errors) == (1, 1, [{"accepted": 1, "errors": ["bad"]}])


def test_send_raises_on_an_error(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "post", lambda url, **kw: _Reply(401, {"message": "unauthorized"}))
    with pytest.raises(RuntimeError, match="401"):
        ps.send([{"id": "1"}], "http://h", "pk", "sk")
