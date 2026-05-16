"""Unit tests for app/ingestion/chunker.py."""
import pytest

from app.ingestion.chunker import TextChunk, chunk_text


def _make_pages(n_words_per_page: int, n_pages: int) -> list[str]:
    word = "token"
    page = " ".join([word] * n_words_per_page)
    return [page] * n_pages


# ── Basic behaviour ───────────────────────────────────────────────────────────

def test_chunk_text_returns_list():
    pages = ["Hello world. " * 10]
    chunks = chunk_text(pages)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_text_indices_sequential():
    pages = _make_pages(n_words_per_page=200, n_pages=4)
    chunks = chunk_text(pages, chunk_size=128, overlap=16)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_chunk_text_respects_size():
    """No chunk should exceed chunk_size tokens by more than one token (encoding rounding)."""
    pages = _make_pages(n_words_per_page=300, n_pages=2)
    chunks = chunk_text(pages, chunk_size=64, overlap=8)
    for c in chunks:
        assert c.token_count <= 64


def test_chunk_text_overlap_produces_more_chunks():
    pages = _make_pages(n_words_per_page=500, n_pages=1)
    chunks_no_overlap = chunk_text(pages, chunk_size=128, overlap=0)
    chunks_overlap = chunk_text(pages, chunk_size=128, overlap=64)
    assert len(chunks_overlap) >= len(chunks_no_overlap)


def test_chunk_text_single_short_page():
    """A very short page should produce exactly one chunk."""
    pages = ["Short text."]
    chunks = chunk_text(pages, chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert "Short text." in chunks[0].text


def test_chunk_text_page_number_tracking():
    """Chunks should record which page their first token came from."""
    pages = ["Page one content " * 50, "Page two content " * 50]
    chunks = chunk_text(pages, chunk_size=64, overlap=0)
    # First chunk must come from page 1
    assert chunks[0].page_number == 1


def test_chunk_text_empty_pages():
    chunks = chunk_text([], chunk_size=512, overlap=50)
    assert chunks == []


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text(["hello"], chunk_size=50, overlap=50)
