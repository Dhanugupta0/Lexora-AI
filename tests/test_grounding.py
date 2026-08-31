from app.rag.grounding import (
    looks_like_refusal, normalize_citations, parse_citations, split_claims,
    strip_invalid_citations, verify_answer,
)


class TestCitationNormalisation:
    def test_converts_full_width_brackets(self):
        assert normalize_citations("Revenue rose【S1】.") == "Revenue rose [S1]."

    def test_converts_parenthesised_citations(self):
        assert parse_citations(normalize_citations("It grew (S2).")) == [2]

    def test_expands_grouped_citations(self):
        assert parse_citations(normalize_citations("Both agree [S1, S2].")) == [1, 2]

    def test_understands_the_source_spelling(self):
        assert parse_citations(normalize_citations("Stated plainly [Source 5].")) == [5]

    def test_leaves_ordinary_parentheticals_alone(self):
        assert parse_citations(normalize_citations("The 2024 report (2024) says so.")) == []

    def test_pulls_a_trailing_citation_inside_the_sentence(self):
        assert normalize_citations("Revenue was $48.2M. [S1]").strip() == "Revenue was $48.2M [S1]."


class TestInvalidCitations:
    def test_removes_out_of_range_markers(self):
        cleaned, invalid = strip_invalid_citations("Real [S1] and fake [S9].", valid_count=2)
        assert invalid == [9] and "[S9]" not in cleaned and "[S1]" in cleaned

    def test_keeps_every_valid_marker(self):
        cleaned, invalid = strip_invalid_citations("A [S1] B [S2].", valid_count=2)
        assert invalid == [] and cleaned.count("[S") == 2


class TestClaimSplitting:
    def test_splits_multi_sentence_answers(self):
        claims = split_claims("Revenue was 48.2 million dollars [S1]. Margin was 74.9 percent [S2].")
        assert len(claims) == 2

    def test_skips_fragments_too_short_to_check(self):
        assert split_claims("Yes. No.") == []

    def test_strips_bullet_markers(self):
        assert split_claims("- Revenue grew by 14.6 percent last year [S1]")[0].startswith("Revenue")


class TestRefusalDetection:
    def test_recognises_an_honest_non_answer(self):
        assert looks_like_refusal("I could not find that in the provided documents.")

    def test_a_cited_answer_is_not_a_refusal(self):
        assert not looks_like_refusal("Revenue was 48.2 million dollars [S1].")

    def test_a_normal_statement_is_not_a_refusal(self):
        assert not looks_like_refusal("The board approved the merger in March.")


class TestVerification:
    PASSAGES = [
        "Total revenue for Q3 FY2024 reached $48.2 million, up 14.6% year over year.",
        "Cost of goods sold was $12.1 million, giving a gross margin of 74.9%.",
    ]

    def test_a_supported_claim_scores_well(self):
        report = verify_answer("Q3 revenue was $48.2 million [S1].", self.PASSAGES)
        assert report.support_ratio > 0.5
        assert report.confidence > 0.4

    def test_a_fabricated_claim_scores_poorly(self):
        report = verify_answer("The CEO is Jane Alvarez and she joined in 2019 [S1].", self.PASSAGES)
        assert report.support_ratio < 0.5

    def test_a_wrong_number_is_penalised(self):
        good = verify_answer("Q3 revenue was $48.2 million [S1].", self.PASSAGES)
        bad = verify_answer("Q3 revenue was $91.7 million [S1].", self.PASSAGES)
        assert bad.claims[0].support < good.claims[0].support

    def test_records_which_sources_were_cited(self):
        report = verify_answer("Margin was 74.9% [S2].", self.PASSAGES)
        assert report.cited_sources == [2]

    def test_no_passages_means_nothing_to_verify(self):
        assert verify_answer("anything at all", []).confidence == 0.0
