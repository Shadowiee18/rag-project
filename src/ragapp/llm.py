"""The answering LLM, in one place: local Ollama, or the BUas LLM server (an OpenAI-compatible chat API).

Both are called the same way: temperature 0 and no "thinking", so answers are repeatable and fast.
Choose with LLM_BACKEND / LLM_MODEL (config.py). Errors are raised, never hidden: if the server does not run the
chosen model, the call fails instead of quietly answering with another one.
"""

import httpx
import ollama

from ragapp import config


def chat(messages: list[dict]) -> str:
    """Send chat messages ([{"role": ..., "content": ...}]) and return the reply text."""
    if config.LLM_BACKEND == "ollama":
        response = ollama.chat(
            model=config.LLM_MODEL,
            messages=messages,
            think=False,  # qwen3 can "think" first; off = faster answers
            options={"temperature": 0, "num_ctx": 8192},  # one num_ctx everywhere, so Ollama doesn't reload the model
        )
        return (response.message.content or "").strip()
    if config.LLM_BACKEND == "buas":
        return _buas_chat(messages)
    raise ValueError(f"Unknown LLM_BACKEND {config.LLM_BACKEND!r}: use 'ollama' or 'buas'")


def _buas_chat(messages: list[dict]) -> str:
    if not (config.BUAS_LLM_API_KEY and config.BUAS_LLM_BASE_URL):
        raise RuntimeError("LLM_BACKEND=buas needs BUAS_LLM_BASE_URL and BUAS_LLM_API_KEY in the .env file")
    response = httpx.post(
        f"{config.BUAS_LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.BUAS_LLM_API_KEY}"},
        json={
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},  # as think=False locally
            "reasoning_effort": "low",  # for models that always reason (from the server's example)
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"BUas LLM server: {response.status_code} {response.text[:300]}")
    return (response.json()["choices"][0]["message"]["content"] or "").strip()
