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


# CALENDAR_FACTS=1 (calendar_facts.py): the Calendar lines are written by code from calendar.yaml and count as a source.
# They already say today's week or break, so with them the Student line needs no source status of its own.
CALENDAR_PROMPT = SYSTEM_PROMPT.replace(
    "- Use ONLY the numbered context passages below. Do not use outside knowledge.",
    "- Use ONLY the numbered context passages below and the Calendar lines. Do not use outside knowledge.",
).replace(
    "- If the context does not contain the answer, reply exactly: I don't know",
    "- If neither the passages nor the Calendar lines contain the answer, reply exactly: I don't know",
).replace(
    '- If a "Student" line is given, use it to understand "today", "this week", "my block" and similar.',
    """- If a "Student" line is given, use it to understand "today", "this week", "my block" and similar.
- The "Calendar" lines come from the official academic calendar, written by code: today's teaching week or break,
  the dates of weeks, days and blocks, holidays and days BUas is closed. They are a source too and need no citation.""",
)


def system_prompt(calendar: bool = False) -> str:
    if calendar:
        return CALENDAR_PROMPT
    return STUDENT_SOURCE_PROMPT if config.STUDENT_LINE_AS_SOURCE else SYSTEM_PROMPT


def build_prompt(question: str, chunks: list[dict], student: str = "", calendar: list[str] = ()) -> str:
    """Number the chunks so the model can cite them as [1], [2], ...

    student: e.g. "Today is Tuesday 24 November 2026. The student is in Year 2, Block B, week 3 (...)".
    calendar: facts from calendar_facts.py, one per line.
    """
    passages = []
    for number, chunk in enumerate(chunks, start=1):
        passages.append(f"[{number}] (page: {chunk['title']})\n{chunk['text']}")
    context = "\n\n".join(passages)
    student_line = f"Student: {student}\n\n" if student else ""
    calendar_lines = "Calendar:\n" + "\n".join(f"- {fact}" for fact in calendar) + "\n\n" if calendar else ""
    return f"Context:\n\n{context}\n\n{student_line}{calendar_lines}Question: {question}"


def generate(question: str, chunks: list[dict], student: str = "", calendar: list[str] = ()) -> str:
    return llm.chat([
        {"role": "system", "content": system_prompt(calendar=bool(calendar))},
        {"role": "user", "content": build_prompt(question, chunks, student, calendar)},
    ])
