import pytest

from app.rag.chunker import (
    Chunk, build_context_header, chunk_blocks, chunk_text, count_tokens, split_sentences,
)
from app.rag.parser import Block


class TestSentenceSplitting:
    def test_splits_on_terminators(self):
        assert len(split_sentences("One thing. Two things! Three things?")) == 3

    def test_keeps_abbreviations_intact(self):
        assert len(split_sentences("Dr. Chen signed it. Then he left.")) == 2

    def test_empty_input_is_safe(self):
        assert split_sentences("") == []


class TestChunking:
    def test_produces_chunks(self):
        assert len(chunk_text(["Hello world. " * 50])) >= 1

    def test_chunk_indices_are_sequential(self):
        chunks = chunk_text(["word " * 300] * 3, chunk_size=128, overlap=16)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_respects_the_token_ceiling(self):
        chunks = chunk_text(["word " * 800], chunk_size=64, overlap=8)
        # overlap is added on top of the budget, so allow one overlap's headroom
        assert all(c.token_count <= 64 + 8 + 16 for c in chunks)

    def test_empty_input_returns_nothing(self):
        assert chunk_text([]) == []

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text(["hello"], chunk_size=50, overlap=50)

    def test_tracks_page_numbers(self):
        chunks = chunk_text(["Page one text. " * 40, "Page two text. " * 40],
                            chunk_size=64, overlap=0)
        assert chunks[0].page_number == 1
        assert any(c.page_number == 2 for c in chunks)

    def test_headings_force_a_chunk_boundary(self):
        blocks = [
            Block("Alpha section body text goes here.", 1, "body", section="Alpha"),
            Block("Beta", 1, "heading", level=2),
            Block("Beta section body text goes here.", 1, "body", section="Beta"),
        ]
        chunks = chunk_blocks(blocks, doc_title="Doc", strategy="recursive")
        assert len({c.section for c in chunks}) == 2


class TestContextualHeaders:
    def test_embed_text_carries_document_and_section(self):
        blocks = [Block("Revenue grew sharply this year.", 3, "body", section="Financials")]
        chunk = chunk_blocks(blocks, doc_title="FY24 Report", strategy="recursive")[0]
        assert "FY24 Report" in chunk.embed_text
        assert "Financials" in chunk.embed_text

    def test_stored_text_stays_clean(self):
        blocks = [Block("Revenue grew sharply this year.", 3, "body", section="Financials")]
        chunk = chunk_blocks(blocks, doc_title="FY24 Report", strategy="recursive")[0]
        assert chunk.text == "Revenue grew sharply this year."
        assert "[Document:" not in chunk.text

    def test_header_includes_the_page_number(self):
        assert "Page: 7" in build_context_header("Doc", "Intro", 7)


class TestTokenCounting:
    def test_counts_grow_with_text_length(self):
        assert count_tokens("a b c d e f g h") > count_tokens("a b")

    def test_empty_string_is_cheap(self):
        assert count_tokens("") <= 1
