"""The answering LLM, in one place: local Ollama, or the BUas LLM server (an OpenAI-compatible chat API).

Both are called the same way: temperature 0 and no "thinking", so answers are repeatable and fast.
Choose with LLM_BACKEND / LLM_MODEL (config.py). Errors are raised, never hidden: if the server does not run the
chosen model, the call fails instead of quietly answering with another one.
"""

import json

import httpx
import ollama

from ragapp import config
from ragapp.tracing import record, tracer


def chat(messages: list[dict]) -> str:
    """Send chat messages ([{"role": ..., "content": ...}]) and return the reply text.

    Traced as one "generation" step: the model and the token counts the backend reports (the prompt and the reply
    only with TRACE_CONTENT=1)."""
    with tracer.start_as_current_span("llm") as span:
        span.set_attribute("langfuse.observation.model.name", config.LLM_MODEL)
        span.set_attribute("gen_ai.request.model", config.LLM_MODEL)
        record(span, "generation", input=messages, content=True, backend=config.LLM_BACKEND)
        if config.LLM_BACKEND == "ollama":
            text, usage = _ollama_chat(messages)
        elif config.LLM_BACKEND == "buas":
            text, usage = _buas_chat(messages)
        else:
            raise ValueError(f"Unknown LLM_BACKEND {config.LLM_BACKEND!r}: use 'ollama' or 'buas'")
        if usage:
            span.set_attribute("langfuse.observation.usage_details", json.dumps(usage))
            span.set_attribute("gen_ai.usage.input_tokens", usage["input"])
            span.set_attribute("gen_ai.usage.output_tokens", usage["output"])
        record(span, output=text, content=True)
        return text


def _usage(prompt_tokens, completion_tokens) -> dict | None:
    """Token counts in Langfuse's keys, or None when the backend reports none."""
    if prompt_tokens is None or completion_tokens is None:
        return None
    return {"input": int(prompt_tokens), "output": int(completion_tokens)}


def _ollama_chat(messages: list[dict]) -> tuple[str, dict | None]:
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=messages,
        think=False,  # qwen3 can "think" first; off = faster answers
        options={"temperature": 0, "num_ctx": 8192},  # one num_ctx everywhere, so Ollama doesn't reload the model
    )
    usage = _usage(getattr(response, "prompt_eval_count", None), getattr(response, "eval_count", None))
    return (response.message.content or "").strip(), usage


def _buas_chat(messages: list[dict]) -> tuple[str, dict | None]:
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
    body = response.json()
    reported = body.get("usage") or {}  # OpenAI style: prompt_tokens / completion_tokens
    usage = _usage(reported.get("prompt_tokens"), reported.get("completion_tokens"))
    return (body["choices"][0]["message"]["content"] or "").strip(), usage
