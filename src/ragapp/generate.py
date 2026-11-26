"""Query time, step 2: give the retrieved chunks to the LLM and get a cited answer."""

from ragapp import config, llm

SYSTEM_PROMPT = """You answer questions from students of the Applied Data Science & AI programme \
at Breda University of Applied Sciences.

Rules:
- Use ONLY the numbered context passages below. Do not use outside knowledge.
- After each claim, cite the passage(s) it comes from, like [1] or [2][3].
- If the context does not contain the answer, reply exactly: I don't know
- If a "Student" line is given, use it to understand "today", "this week", "my block" and similar.
- Be concise: a few sentences, or short steps for "how do I" questions."""

# STUDENT_LINE_AS_SOURCE=1: the same prompt, but the Student line (written by code from the official calendar) counts
# as a source. A larger model followed "ONLY the numbered passages" so strictly that it ignored "this week is the
# Christmas holiday" in the Student line and answered "I don't know".
STUDENT_SOURCE_PROMPT = SYSTEM_PROMPT.replace(
    "- Use ONLY the numbered context passages below. Do not use outside knowledge.",
    "- Use ONLY the numbered context passages below and the Student line. Do not use outside knowledge.",
).replace(
    "- If the context does not contain the answer, reply exactly: I don't know",
    "- If neither the passages nor the Student line contain the answer, reply exactly: I don't know",
).replace(
    '- If a "Student" line is given, use it to understand "today", "this week", "my block" and similar.',
    """- The "Student" line comes from the official academic calendar and is a source too: today's date, the student's
  year, block and week, and whether this week is a holiday or study week. Use it for "today", "this week",
  "my block" and similar; facts from it need no citation.""",
)


def system_prompt() -> str:
    return STUDENT_SOURCE_PROMPT if config.STUDENT_LINE_AS_SOURCE else SYSTEM_PROMPT


def build_prompt(question: str, chunks: list[dict], student: str = "") -> str:
    """Number the chunks so the model can cite them as [1], [2], ...

    student: e.g. "Today is Tuesday 24 November 2026. The student is in Year 2, Block B, week 3 (...)".
    """
    passages = []
    for number, chunk in enumerate(chunks, start=1):
        passages.append(f"[{number}] (page: {chunk['title']})\n{chunk['text']}")
    context = "\n\n".join(passages)
    student_line = f"Student: {student}\n\n" if student else ""
    return f"Context:\n\n{context}\n\n{student_line}Question: {question}"


def generate(question: str, chunks: list[dict], student: str = "") -> str:
    return llm.chat([
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": build_prompt(question, chunks, student)},
    ])
