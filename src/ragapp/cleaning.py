"""Corpus cleaning rules, decided from a corpus-only EDA (analysis/corpus_eda.py), never from eval failures.

Every rule is computed from the page itself, so it applies to any number of pages:
1. Drop near-empty pages: under MIN_CHARS characters. In this corpus nothing below 200 is real content
   (test stubs, index pages holding only a heading); between 200 and 1000 real pages start to appear.
2. Collapse EXACT duplicates only (same text after whitespace normalisation). Near-duplicates are kept:
   the EDA found they differ in real facts (e.g. Semester 1 vs 2 of the same project: "4 phases" vs "3").
3. Tag template pages (page_type="template") instead of deleting them, so search can leave them out for students.
"""

import hashlib
import re
from dataclasses import dataclass, field

from ragapp.documents import Document

MIN_CHARS = 200

# Author instructions and fill-in fields seen in this site's templates.
TEMPLATE_SIGNALS = re.compile(
    r"this is a template|write (?:an|the|your) [^.]{0,40} here|fixed text for all blocks|\b20YY\b|\bShortName\b|\bLongName\b|\(position\)",
    re.IGNORECASE,
)


@dataclass
class CleaningReport:
    kept: list[Document] = field(default_factory=list)
    dropped_short: list[str] = field(default_factory=list)
    duplicate_of: dict[str, str] = field(default_factory=dict)  # removed copy id -> kept page id
    templates: list[str] = field(default_factory=list)


def prose_words(text: str) -> int:
    """Words of real prose: not headings, not table rows, not link text."""
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#") and "|" not in line]
    lines = [re.sub(r"\[[^\]]*\]\([^)]*\)", "", line) for line in lines]
    return len(re.findall(r"[A-Za-z]{2,}", " ".join(lines)))


def is_template(doc: Document) -> bool:
    """In the site's own Templates section, or 2+ author instructions, or headings with no prose at all."""
    if doc.metadata.get("section") == "Templates":
        return True
    if len(TEMPLATE_SIGNALS.findall(doc.text)) >= 2:
        return True
    return len(doc.text) >= MIN_CHARS and prose_words(doc.text) == 0


def text_fingerprint(text: str) -> str:
    """Same words in the same order = same page (ignores case, punctuation and link syntax),
    matching the EDA's definition of an exact duplicate (word-sequence overlap of 1.0)."""
    words = re.findall(r"\w+", text.lower())
    return hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()


def canonical(ids: list[str]) -> str:
    """Which copy to keep: the shortest path (fewest folders), then alphabetical, so the choice is stable."""
    return min(ids, key=lambda i: (i.count("/"), len(i), i))


def clean(docs: list[Document]) -> CleaningReport:
    report = CleaningReport()
    long_enough = []
    for doc in docs:
        if len(doc.text) < MIN_CHARS:
            report.dropped_short.append(doc.id)
        else:
            long_enough.append(doc)

    groups: dict[str, list[Document]] = {}
    for doc in long_enough:
        groups.setdefault(text_fingerprint(doc.text), []).append(doc)

    for copies in groups.values():
        keep_id = canonical([d.id for d in copies])
        keep = next(d for d in copies if d.id == keep_id)
        others = [d for d in copies if d.id != keep_id]
        for d in others:
            report.duplicate_of[d.id] = keep_id
        metadata = dict(keep.metadata)
        metadata["page_type"] = "template" if is_template(keep) else "content"
        if others:
            metadata["also_at"] = " ".join(d.source for d in others)  # metadata values are strings
            # The merged page belongs to every year/block its copies were in ("BlockA BlockB"), so the
            # year/block search filter still finds it from each. A copy with no year/block is shared by all.
            # (Rough edge: copies in Year1/BlockA and Year2/BlockB would also match Year1/BlockB.)
            for key in ("year", "block"):
                values = {d.metadata.get(key, "") for d in copies}
                metadata[key] = "" if "" in values else " ".join(sorted(values))
        report.kept.append(keep.model_copy(update={"metadata": metadata}))
        if metadata["page_type"] == "template":
            report.templates.append(keep_id)

    report.kept.sort(key=lambda d: d.id)
    return report
