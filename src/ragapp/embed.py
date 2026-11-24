"""Turn text into vectors with the local Ollama embedding model."""

import ollama

from ragapp import config


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per text, in the same order."""
    response = ollama.embed(model=config.EMBED_MODEL, input=texts)
    return response.embeddings
