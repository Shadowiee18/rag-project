"""Exploratory data analysis of the corpus ONLY: no eval questions, no eval results.

Cleaning rules must come from properties of the data (so they'd work on 5 million pages),
not from looking at which test questions failed (that would be overfitting to the eval).

Run:  uv run python analysis/corpus_eda.py
Writes analysis/output/corpus_eda.json (git-ignored: it names course pages).
"""

import json
import re
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from ragapp import config
from ragapp.documents import load_documents
from ragapp.store import get_client

OUT = Path(__file__).resolve().parent / "output"
SHINGLE = 8  # words per shingle for near-duplicate detection
PLACEHOLDER = re.compile(
    r"\bTBD\b|\bTBA\b|\blorem ipsum\b|\bplaceholder\b|\[insert[^\]]*\]|<insert[^>]*>|\bxx[.:/]xx\b|\bdd[.:/]mm\b|\.\.\.\s*\|",
    re.IGNORECASE,
)
YEAR = re.compile(r"\b(20[12]\d)\b")


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def page_stats(docs) -> dict:
    lengths = sorted(len(d.text) for d in docs)
    q = lambda p: lengths[int(p * (len(lengths) - 1))]
    by_section = Counter(d.metadata.get("section", "") for d in docs)
    return {
        "pages": len(docs),
        "chars": {"min": lengths[0], "p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": lengths[-1]},
        "under_200_chars": [d.id for d in docs if len(d.text) < 200],
        "pages_per_section": dict(by_section.most_common()),
    }


def near_duplicates(docs, threshold: float = 0.8) -> list[dict]:
    """Pairs of pages whose 8-word sequences overlap a lot (Jaccard similarity)."""
    sh = {d.id: shingles(d.text) for d in docs}
    pairs = []
    for a, b in combinations([d.id for d in docs if len(sh[d.id]) >= 20], 2):
        inter = len(sh[a] & sh[b])
        if inter:
            jaccard = inter / len(sh[a] | sh[b])
            if jaccard >= threshold:
                pairs.append({"a": a, "b": b, "jaccard": round(jaccard, 3)})
    return sorted(pairs, key=lambda p: -p["jaccard"])


def placeholder_density(docs) -> list[dict]:
    """Placeholder hits per 1,000 characters: template-like pages score high."""
    rows = []
    for d in docs:
        hits = len(PLACEHOLDER.findall(d.text))
        if hits:
            rows.append({"id": d.id, "hits": hits, "per_1000_chars": round(1000 * hits / max(len(d.text), 1), 2)})
    return sorted(rows, key=lambda r: -r["per_1000_chars"])


def year_mentions(docs) -> dict:
    latest = Counter()
    for d in docs:
        years = [int(y) for y in YEAR.findall(d.text)]
        latest[max(years) if years else "none"] += 1
    return {str(k): v for k, v in sorted(latest.items(), key=lambda kv: str(kv[0]))}


def hubness(k: int = 10) -> dict:
    """k-occurrence: how often each chunk is among the k nearest neighbours of OTHER chunks.

    In a healthy space most chunks appear about k times. "Hubs" appear far more often: they are
    near everything, so they also show up for vague queries. Computed from the vectors alone.
    """
    client = get_client()
    points, offset = [], None
    while True:
        batch, offset = client.scroll(config.COLLECTION, limit=1000, offset=offset, with_vectors=True, with_payload=True)
        points += batch
        if offset is None:
            break
    vectors = np.array([p.vector for p in points], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)
    neighbours = np.argpartition(-sims, k, axis=1)[:, :k]
    counts = np.bincount(neighbours.ravel(), minlength=len(points))

    mean, std = counts.mean(), counts.std()
    skewness = float(((counts - mean) ** 3).mean() / std**3)
    order = np.argsort(-counts)
    top = [
        {"doc_id": points[i].payload["doc_id"], "chunk": points[i].payload["chunk_index"], "k_occurrence": int(counts[i])}
        for i in order[:20]
    ]
    per_page = Counter()
    for i, c in enumerate(counts):
        per_page[points[i].payload["doc_id"]] += int(c)
    return {
        "chunks": len(points),
        "k": k,
        "k_occurrence": {"mean": round(float(mean), 2), "median": float(np.median(counts)), "max": int(counts.max()),
                         "never_a_neighbour": int((counts == 0).sum())},
        "skewness": round(skewness, 2),
        "chunks_with_k_occurrence_over_5k": int((counts > 5 * k).sum()),
        "top_hub_chunks": top,
        "top_hub_pages_total": [{"doc_id": d, "k_occurrence_sum": n} for d, n in per_page.most_common(10)],
    }


def main() -> None:
    docs = load_documents(config.CORPUS_PATH)
    report = {
        "pages": page_stats(docs),
        "near_duplicates_jaccard_0.8": near_duplicates(docs),
        "placeholder_density": placeholder_density(docs)[:15],
        "latest_year_mentioned_per_page": year_mentions(docs),
        "hubness": hubness(),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "corpus_eda.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
