# Judge rubric (v1.1)

Changes in v1.1: key facts are defined by what the question asks (two graders applied v1 differently).

Used by every judge, human or model, so grades are comparable. Grade each answer on its own,
using only: the question, the expected answer, the numbered context chunks the system saw, and the answer.
Do not use outside knowledge about the course.

## 1. Correctness: compared with the expected answer
- `correct`: contains every key fact of the expected answer, and nothing that contradicts it.
- `partial`: some key facts right, but at least one missing; nothing contradicts the expected answer.
- `wrong`: contradicts the expected answer, answers a different question, or wrongly says "I don't know".

Key facts are the specific things the question asks for: dates, times, numbers, names, steps, rules.
Details in the expected answer that the question does not ask about (a reason, an example, a downside,
a next step) are optional: leaving them out does not make an answer `partial`.
Extra correct detail is fine. Different wording is fine.

## 2. Faithfulness: compared with the context chunks
- `faithful`: every factual claim in the answer is supported by the context chunks.
- `unfaithful`: at least one claim is not in the chunks (made up, or taken from general knowledge),
  or a claim is applied to the wrong thing (e.g. Block B's fact presented as Block A's).
  List the unsupported claims.

An answer can be faithful but wrong (the chunks were wrong) and correct but unfaithful (lucky guess).

## 3. Citations
- `ok`: every claim cites a chunk, and each cited chunk actually contains that claim.
- `bad`: missing citations, or a citation points to a chunk that doesn't support the claim.
- `n/a`: the answer is "I don't know".

## 4. Unanswerable questions (expected answer "I don't know", no source pages)
- `correct`: the answer says it doesn't know (it may add what the context *does* say, as long as it doesn't
  claim to answer the question).
- `wrong`: it gives an answer to the question.
Faithfulness and citations are still graded if it makes claims.

## Output: one JSON object per question
{"id": "a01", "correctness": "correct|partial|wrong", "faithfulness": "faithful|unfaithful",
 "unsupported_claims": ["..."], "citations": "ok|bad|n/a", "reason": "one or two sentences"}
