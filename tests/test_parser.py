import pytest

from app.rag.parser import Block, clean_text, parse_document


class TestTextHygiene:
    def test_rejoins_hyphenated_line_breaks(self):
        assert "hyphenation" in clean_text("hyphen-\nation")

    def test_collapses_runs_of_blank_lines(self):
        assert "\n\n\n" not in clean_text("a\n\n\n\n\nb")

    def test_strips_control_characters(self):
        assert "\x07" not in clean_text("a\x07b")


class TestPlainText:
    def test_parses_basic_text(self):
        doc = parse_document(b"Hello world\nLine two", "txt")
        assert doc.page_count >= 1
        assert "Hello world" in doc.blocks[0].text

    def test_empty_file_raises(self):
        with pytest.raises(ValueError, match="empty"):
            parse_document(b"   ", "txt")

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document(b"data", "xlsx")

    def test_zero_bytes_raises(self):
        with pytest.raises(ValueError):
            parse_document(b"", "txt")

    def test_latin1_fallback_decoding(self):
        doc = parse_document("Ça va bien aujourd'hui".encode("latin-1"), "txt")
        assert doc.blocks

    def test_long_text_spans_multiple_pages(self):
        doc = parse_document(("Line of text here.\n" * 400).encode(), "txt")
        assert doc.page_count >= 2


class TestMarkdownStructure:
    SAMPLE = b"""# Annual Report

## Revenue
Revenue reached ten million dollars.

## Risks
Concentration risk remains high.
"""

    def test_detects_headings(self):
        doc = parse_document(self.SAMPLE, "md")
        headings = [b.text for b in doc.blocks if b.kind == "heading"]
        assert "Revenue" in headings and "Risks" in headings

    def test_title_comes_from_first_heading(self):
        assert parse_document(self.SAMPLE, "md").title == "Annual Report"

    def test_body_blocks_carry_their_section_path(self):
        doc = parse_document(self.SAMPLE, "md")
        revenue = next(b for b in doc.blocks if "ten million" in b.text)
        assert revenue.section.endswith("Revenue")

    def test_section_path_is_not_duplicated(self):
        doc = parse_document(self.SAMPLE, "md")
        revenue = next(b for b in doc.blocks if "ten million" in b.text)
        assert "Revenue > Revenue" not in revenue.section

    def test_filename_supplies_a_missing_title(self):
        doc = parse_document(b"just some prose without any heading at all", "txt",
                             filename="quarterly_notes.txt")
        assert doc.title == "quarterly notes"
