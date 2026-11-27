"""Turn a short, vague student question into better search queries, before retrieval.

v1 (rewrite_query): the LLM writes one query, including the student's year/block/week. REJECTED by the eval:
    the 8B model copied "Block B" from its examples into most queries and dropped key terms (page-style hit@5 93% -> 61%).
v2 (search_queries): the LLM only expands vague words; the CODE adds year/block/week, and only for time
    questions; the original question is always searched too, and the results are merged (multi-query + RRF).

The original question is still what the answering LLM sees; only the search uses the rewrites.
"""

import re

from ragapp import llm
from ragapp.resolver import block_ref_in_question
from ragapp.retrieve import year_block_in_question
from ragapp.student import StudentContext, search_phrase

REWRITE_PROMPT = """You turn a student's question into a search query for their study programme's course website.

{context}

Rules:
- Keep every specific term from the question (task numbers, tool names, course names).
- Spell out vague words: "llm" -> "generative AI / LLM tools", "citing" -> "citation style / referencing".
- Never add task numbers, tools or topics the question does not mention.
{context_rules}
- If the question names a year or block itself, keep that one.
- Output only the search query, one line, no explanation.

Examples:
{examples}

Question: {question}
Search query:"""

WITH_CONTEXT_RULES = """- Add the student's year and block from the context when the question is about their programme
  (tasks, deadlines, rules, deliverables, schedule) and names no block itself.
- Add the week only when the question is about time ("today", "this week", "next", "left", "still have")."""
WITH_CONTEXT_EXAMPLES = """whats due this week -> Year 2 Block B week 3 deadlines deliverables due
deadline task 8 -> Year 2 Block B DataLab Task 8 deadline
is using llm allowed -> Year 2 Block B rules use of generative AI LLM tools allowed
vpn? -> how to set up VPN access FortiClient off campus
what is overfitting -> overfitting definition machine learning"""

NO_CONTEXT_RULES = "- There is no student context: never add a year, block or week."
NO_CONTEXT_EXAMPLES = """deadline task 8 -> DataLab Task 8 deadline
is using llm allowed -> rules use of generative AI LLM tools allowed
vpn? -> how to set up VPN access FortiClient off campus
what is overfitting -> overfitting definition machine learning"""


def rewrite_query(question: str, context: str) -> str:
    """context: describe(ctx, with_plan=False), or "" if we don't know who is asking.

    If the question names a year or block itself, the student context is dropped, so the model
    can't replace "Year 2 Block A" with the student's own block (it did, when given the chance).
    """
    if any(year_block_in_question(question)):
        context = ""
    prompt = REWRITE_PROMPT.format(
        context=context or "No student context.",
        context_rules=WITH_CONTEXT_RULES if context else NO_CONTEXT_RULES,
        examples=WITH_CONTEXT_EXAMPLES if context else NO_CONTEXT_EXAMPLES,
        question=question,
    )
    text = llm.chat([{"role": "user", "content": prompt}])
    query = text.splitlines()[0].strip() if text else ""
    return query or question


# ---------------------------------------------------------------- v2

EXPAND_PROMPT = """Rewrite a student's question as a search query for a course website.

Rules:
- Keep every word of the question that carries meaning.
- Replace vague words or abbreviations with what they likely mean, and add one or two synonyms.
- Do NOT add years, blocks, weeks, dates, task numbers, course names or tools that are not in the question.
- Output only the search query, one line.

Examples:
is using llm allowed -> is using generative AI LLM tools such as ChatGPT allowed rules
which citing do i have to use -> which citation style referencing format must I use
vpn? -> VPN access setup connect from home
deadline task 8 -> task 8 deadline due date
keras backend -> Keras backend TensorFlow JAX PyTorch

Question: {question}
Search query:"""

# Time words: the question depends on today's date, so the student's week matters for the search.
TIME_WORDS = re.compile(
    r"\b(today|tonight|tomorrow|this week|next week|this month|left|remaining|still have|upcoming|now|due|deadlines?|next)\b",
    re.IGNORECASE,
)
# Anything that looks like a year/block/week the LLM was told not to add.
CONTEXT_WORDS = re.compile(r"\b(year\s*\d|block\s*[a-d]|semester\s*\d|week\s*\d+|w\d+)\b", re.IGNORECASE)


def expand_query(question: str) -> str:
    """LLM: expand vague words only. Code: remove any year/block/week the model adds anyway."""
    text = llm.chat([{"role": "user", "content": EXPAND_PROMPT.format(question=question)}])
    expanded = text.splitlines()[0].strip() if text else ""
    if not CONTEXT_WORDS.search(question):
        expanded = " ".join(CONTEXT_WORDS.sub(" ", expanded).split())
    return expanded or question


def search_queries(question: str, student: StudentContext | None) -> list[str]:
    """The queries to search with: [original, expanded], plus the student's year/block/week for time questions.

    The original is always included, so a bad expansion can only add candidates, never remove the good ones.
    """
    queries = [question, expand_query(question)]
    names_year_or_block = any(year_block_in_question(question)) or block_ref_in_question(question)
    if student and not names_year_or_block and TIME_WORDS.search(question):
        phrase = search_phrase(student)
        if phrase:
            queries = [f"{phrase} {q}" for q in queries]
    return list(dict.fromkeys(queries))  # drop exact duplicates, keep order


def week_queries(question: str, student: StudentContext | None) -> list[str]:
    """QUERY_REWRITE=3, no LLM: the original question, plus a copy with the student's year/block/week for
    time questions ("whats due this week" -> "Year 2 Block B week 3 whats due this week")."""
    queries = [question]
    names_year_or_block = any(year_block_in_question(question)) or block_ref_in_question(question)
    if student and not names_year_or_block and TIME_WORDS.search(question):
        phrase = search_phrase(student)
        if phrase:
            queries.append(f"{phrase} {question}")
    return queries
