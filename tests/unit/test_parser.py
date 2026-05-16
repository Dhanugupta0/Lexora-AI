"""Unit tests for app/ingestion/parser.py."""
import io

import pytest

from app.ingestion.parser import ParserError, ParsedDocument, parse_document


# ── TXT ───────────────────────────────────────────────────────────────────────

def test_parse_txt_basic():
    text = b"Line one\nLine two\nLine three"
    result = parse_document(text, "txt")
    assert isinstance(result, ParsedDocument)
    assert result.page_count >= 1
    assert "Line one" in result.full_text


def test_parse_txt_empty_raises():
    with pytest.raises(ParserError, match="empty"):
        parse_document(b"   \n   ", "txt")


def test_parse_txt_virtual_pages():
    """50+ lines should produce at least 2 virtual pages."""
    lines = "\n".join(f"Line {i}" for i in range(110))
    result = parse_document(lines.encode(), "txt")
    assert result.page_count >= 2


def test_parse_txt_latin1_fallback():
    latin1_bytes = "Ça va\nIñtërnâtiônàlizætiøn".encode("latin-1")
    result = parse_document(latin1_bytes, "txt")
    assert result.page_count >= 1


# ── PDF ───────────────────────────────────────────────────────────────────────

def test_parse_pdf_valid(sample_pdf_bytes):
    """Smoke test — a minimal PDF should not raise."""
    try:
        result = parse_document(sample_pdf_bytes, "pdf")
        assert result.page_count >= 1
    except ParserError as e:
        # Acceptable if the minimal PDF has no extractable text layer
        assert "no extractable text" in str(e).lower()


def test_parse_pdf_invalid_bytes():
    with pytest.raises(ParserError):
        parse_document(b"not a pdf at all", "pdf")


# ── Unsupported ───────────────────────────────────────────────────────────────

def test_parse_unsupported_type():
    with pytest.raises(ParserError, match="Unsupported"):
        parse_document(b"data", "xlsx")


def test_parse_extension_with_dot():
    """file_type may be passed with a leading dot."""
    result = parse_document(b"Hello world", ".txt")
    assert result.page_count >= 1
