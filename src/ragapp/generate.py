"""Query time, step 2: give the retrieved chunks to the LLM and get a cited answer."""

import ollama

from ragapp import config

SYSTEM_PROMPT = """You answer questions from students of the Applied Data Science & AI programme \
at Breda University of Applied Sciences.

Rules:
- Use ONLY the numbered context passages below. Do not use outside knowledge.
- After each claim, cite the passage(s) it comes from, like [1] or [2][3].
- If the context does not contain the answer, reply exactly: I don't know
- Be concise: a few sentences, or short steps for "how do I" questions."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Number the chunks so the model can cite them as [1], [2], ..."""
    passages = []
    for number, chunk in enumerate(chunks, start=1):
        passages.append(f"[{number}] (page: {chunk['title']})\n{chunk['text']}")
    context = "\n\n".join(passages)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def generate(question: str, chunks: list[dict]) -> str:
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        think=False,  # qwen3 can "think" first; off = faster answers
        options={"temperature": 0, "num_ctx": 8192},
    )
    return response.message.content.strip()
