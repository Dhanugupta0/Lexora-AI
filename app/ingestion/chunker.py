"""Sliding-window text chunker with tiktoken tokenisation."""
import logging
from dataclasses import dataclass
from typing import List, Optional

import tiktoken

from app.config import get_settings

logger = logging.getLogger(__name__)

# cl100k_base is used by GPT-4, text-embedding-3-small, etc.
_ENCODING_NAME = "cl100k_base"


@dataclass
class TextChunk:
    text: str
    token_count: int
    chunk_index: int
    page_number: Optional[int] = None  # 1-based page index from the source document


def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def chunk_text(
    pages: List[str],
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[TextChunk]:
    """Split a list of page strings into overlapping token-bounded chunks.

    The chunker walks through all pages sequentially, building a token buffer.
    When the buffer exceeds *chunk_size* it emits a chunk and slides forward by
    (chunk_size - overlap) tokens before continuing.

    Args:
        pages: List of page strings (output of the parser).
        chunk_size: Max tokens per chunk. Defaults to ``CHUNK_SIZE_TOKENS``.
        overlap: Token overlap between consecutive chunks. Defaults to ``CHUNK_OVERLAP_TOKENS``.

    Returns:
        Ordered list of :class:`TextChunk` objects.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
    overlap = overlap or settings.CHUNK_OVERLAP_TOKENS

    enc = get_encoder()
    chunks: List[TextChunk] = []
    token_buffer: List[int] = []
    page_map: List[int] = []  # parallel list: page index for each token
    chunk_index = 0

    for page_idx, page_text in enumerate(pages):
        page_tokens = enc.encode(page_text)
        token_buffer.extend(page_tokens)
        page_map.extend([page_idx + 1] * len(page_tokens))  # 1-based

    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    start = 0
    while start < len(token_buffer):
        end = min(start + chunk_size, len(token_buffer))
        token_slice = token_buffer[start:end]
        text = enc.decode(token_slice)
        # Use the page number of the first token in this slice
        page_num = page_map[start] if page_map else None

        chunks.append(
            TextChunk(
                text=text,
                token_count=len(token_slice),
                chunk_index=chunk_index,
                page_number=page_num,
            )
        )
        chunk_index += 1

        if end == len(token_buffer):
            break
        start += stride

    logger.debug("Chunked %d pages → %d chunks", len(pages), len(chunks))
    return chunks
