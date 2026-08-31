"""Blocks -> retrieval-ready chunks.

The naive approach -- slice the token stream every N tokens -- cuts sentences in
half and mixes unrelated topics into one vector, which is the single biggest
source of bad retrieval. Two things fix that here:

1. **Semantic chunking.** Sentences are embedded and split where consecutive
   similarity drops below a percentile threshold, so a boundary lands where the
   topic actually changes. Headings are always hard boundaries.

2. **Contextual headers.** Each chunk is *embedded* with its document title and
   heading breadcrumb prepended, while the *stored* text stays clean. A chunk
   reading "it grew 14% year over year" becomes findable by "revenue growth"
   because its embedded form carries `[Document: FY24 Report > Revenue]`.

Chunks are then token-bounded: oversized ones are split, undersized ones merged
into their neighbour, and a configurable sentence overlap is stitched back on so
no answer falls through a seam.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.config import get_settings
from app.rag.parser import Block

logger = logging.getLogger(__name__)
settings = get_settings()

_ENCODER = None


def _encoder():
    """cl100k_base is a good universal token proxy; fall back to word counts."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:                                    # noqa: BLE001
            logger.warning("tiktoken unavailable -- approximating tokens from words")
            _ENCODER = False
    return _ENCODER


def count_tokens(text: str) -> int:
    enc = _encoder()
    if enc:
        return len(enc.encode(text))
    return max(1, int(len(text.split()) * 1.3))


def _truncate_tokens(text: str, limit: int) -> str:
    enc = _encoder()
    if enc:
        tokens = enc.encode(text)
        return text if len(tokens) <= limit else enc.decode(tokens[:limit])
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit])


@dataclass
class Chunk:
    text: str                       # clean text, shown to the user and the LLM
    embed_text: str                 # text actually embedded (may carry a context header)
    token_count: int
    chunk_index: int
    page_number: int
    section: str = ""
    doc_title: str = ""
    kind: str = "body"
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")

# Trailing tokens that end in a period without ending a sentence. Rather than
# fight Python's fixed-width lookbehind, we split first and re-join fragments
# whose last word is one of these.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "no", "vs", "fig", "eq",
    "al", "etc", "inc", "ltd", "co", "corp", "dept", "est", "approx", "e.g",
    "i.e", "cf", "vol", "ch", "sec", "p", "pp",
}
_TRAILING_WORD = re.compile(r"([A-Za-z.]+)\.[\"')\]]*$")


def _ends_with_abbreviation(fragment: str) -> bool:
    match = _TRAILING_WORD.search(fragment.strip())
    if not match:
        return False
    word = match.group(1).lower().rstrip(".")
    # A lone capital is almost always an initial ("J. R. Smith").
    return word in _ABBREVIATIONS or len(word) == 1


def split_sentences(text: str) -> List[str]:
    """Regex sentence splitter -- no NLTK/spaCy model download at runtime."""
    sentences: List[str] = []
    for paragraph in re.split(r"\n{2,}", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for piece in _SENTENCE_END.split(paragraph):
            piece = piece.strip()
            if not piece:
                continue
            if sentences and _ends_with_abbreviation(sentences[-1]):
                sentences[-1] = f"{sentences[-1]} {piece}"
            else:
                sentences.append(piece)
    return sentences or ([text.strip()] if text.strip() else [])


# --------------------------------------------------------------------------- #
# Grouping helpers
# --------------------------------------------------------------------------- #
@dataclass
class _Unit:
    """A sentence plus the block context it came from."""
    text: str
    page: int
    section: str
    kind: str
    hard_break: bool = False        # a heading or table starts a new chunk


def _blocks_to_units(blocks: Sequence[Block]) -> List[_Unit]:
    units: List[_Unit] = []
    # The parser already stamps every body block with its full heading path, so
    # we only track headings here to know where a chunk must be forced to start.
    after_heading = False

    for block in blocks:
        if block.kind == "heading":
            after_heading = True
            continue

        if block.kind == "table":
            units.append(_Unit(block.text, block.page, block.section, "table", hard_break=True))
            after_heading = False
            continue

        for i, sentence in enumerate(split_sentences(block.text)):
            units.append(_Unit(sentence, block.page, block.section, block.kind,
                               hard_break=(i == 0 and after_heading)))
        after_heading = False

    return units


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    low, high = math.floor(k), math.ceil(k)
    if low == high:
        return ordered[int(k)]
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def _semantic_groups(units: List[_Unit], max_tokens: int) -> List[List[_Unit]]:
    """Break where consecutive-sentence similarity falls off a cliff."""
    if len(units) < 3:
        return [units] if units else []

    try:
        from app.rag.embedder import get_embedder
        vectors = get_embedder().embed([u.text for u in units], "similarity")
    except Exception as exc:                                 # noqa: BLE001
        logger.warning("Semantic chunking unavailable (%s) -- using structural grouping", exc)
        return _structural_groups(units, max_tokens)

    distances = [1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(units) - 1)]
    threshold = _percentile(distances, settings.SEMANTIC_BREAKPOINT_PERCENTILE)

    groups: List[List[_Unit]] = []
    current: List[_Unit] = [units[0]]
    tokens = count_tokens(units[0].text)

    for i in range(1, len(units)):
        unit = units[i]
        unit_tokens = count_tokens(unit.text)
        topic_shift = distances[i - 1] > threshold
        section_shift = unit.section != current[-1].section

        if unit.hard_break or section_shift or tokens + unit_tokens > max_tokens or (
            topic_shift and tokens >= settings.CHUNK_MIN_TOKENS
        ):
            groups.append(current)
            current, tokens = [unit], unit_tokens
        else:
            current.append(unit)
            tokens += unit_tokens

    if current:
        groups.append(current)
    return groups


def _structural_groups(units: List[_Unit], max_tokens: int) -> List[List[_Unit]]:
    """Cheap fallback: pack sentences up to the limit, respecting hard breaks."""
    groups: List[List[_Unit]] = []
    current: List[_Unit] = []
    tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit.text)
        if current and (unit.hard_break or unit.section != current[-1].section
                        or tokens + unit_tokens > max_tokens):
            groups.append(current)
            current, tokens = [], 0
        current.append(unit)
        tokens += unit_tokens

    if current:
        groups.append(current)
    return groups


def _apply_token_bounds(groups: List[List[_Unit]], max_tokens: int, min_tokens: int) -> List[List[_Unit]]:
    """Split anything oversized, absorb anything too small into its neighbour."""
    sized: List[List[_Unit]] = []
    for group in groups:
        buffer: List[_Unit] = []
        tokens = 0
        for unit in group:
            unit_tokens = count_tokens(unit.text)
            if unit_tokens > max_tokens:                     # one giant sentence / table
                if buffer:
                    sized.append(buffer)
                    buffer, tokens = [], 0
                sized.extend([_Unit(part, unit.page, unit.section, unit.kind)]
                             for part in _hard_split(unit.text, max_tokens))
                continue
            if tokens + unit_tokens > max_tokens and buffer:
                sized.append(buffer)
                buffer, tokens = [], 0
            buffer.append(unit)
            tokens += unit_tokens
        if buffer:
            sized.append(buffer)

    merged: List[List[_Unit]] = []
    for group in sized:
        tokens = sum(count_tokens(u.text) for u in group)
        if (
            merged
            and tokens < min_tokens
            and group[0].section == merged[-1][-1].section
            and sum(count_tokens(u.text) for u in merged[-1]) + tokens <= max_tokens
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


def _hard_split(text: str, max_tokens: int) -> List[str]:
    enc = _encoder()
    if not enc:
        words = text.split()
        step = max(1, max_tokens)
        return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]
    tokens = enc.encode(text)
    return [enc.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def _overlap_tail(units: List[_Unit], overlap_tokens: int) -> List[_Unit]:
    """Trailing sentences of a group, up to the overlap budget."""
    tail: List[_Unit] = []
    budget = 0
    for unit in reversed(units):
        unit_tokens = count_tokens(unit.text)
        if budget + unit_tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        budget += unit_tokens
    return tail


def build_context_header(doc_title: str, section: str, page: int) -> str:
    parts = []
    if doc_title:
        parts.append(f"Document: {_truncate_tokens(doc_title, 24)}")
    if section:
        parts.append(f"Section: {_truncate_tokens(section, 40)}")
    parts.append(f"Page: {page}")
    return "[" + " | ".join(parts) + "]"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def chunk_blocks(
    blocks: Sequence[Block],
    *,
    doc_title: str = "",
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    strategy: Optional[str] = None,
) -> List[Chunk]:
    """Turn parsed blocks into overlapping, context-headed chunks."""
    chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
    overlap = settings.CHUNK_OVERLAP_TOKENS if overlap is None else overlap
    strategy = (strategy or settings.CHUNK_STRATEGY).lower()

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    units = _blocks_to_units(blocks)
    if not units:
        return []

    groups = (
        _semantic_groups(units, chunk_size)
        if strategy == "semantic"
        else _structural_groups(units, chunk_size)
    )
    groups = _apply_token_bounds(groups, chunk_size, settings.CHUNK_MIN_TOKENS)

    chunks: List[Chunk] = []
    previous: List[_Unit] = []

    for group in groups:
        if not group:
            continue
        carried = _overlap_tail(previous, overlap) if (previous and overlap > 0) else []
        # don't carry context across a section boundary -- it would be misleading
        if carried and carried[-1].section != group[0].section:
            carried = []

        body = " ".join(u.text for u in carried + group).strip()
        if not body:
            continue

        section = group[0].section
        page = group[0].page
        header = build_context_header(doc_title, section, page)
        embed_text = f"{header}\n{body}" if settings.CONTEXTUAL_HEADERS else body

        chunks.append(Chunk(
            text=body,
            embed_text=embed_text,
            token_count=count_tokens(body),
            chunk_index=len(chunks),
            page_number=page,
            section=section,
            doc_title=doc_title,
            kind=group[0].kind,
            meta={"overlap_sentences": len(carried), "sentences": len(group)},
        ))
        previous = group

    return chunks


def chunk_text(
    pages: Sequence[str],
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    doc_title: str = "",
) -> List[Chunk]:
    """Convenience wrapper for plain page strings (used by tests and scripts)."""
    blocks = [Block(text=text, page=i) for i, text in enumerate(pages, start=1) if text and text.strip()]
    return chunk_blocks(blocks, doc_title=doc_title, chunk_size=chunk_size, overlap=overlap)
