from app.rag.keyword import tokenize
from app.rag.reranker import Reranker
from app.rag.retriever import RetrievedChunk, maximal_marginal_relevance, reciprocal_rank_fusion


def chunk(cid, score=0.5, text="some text"):
    return RetrievedChunk(chunk_id=cid, document_id="d1", chunk_index=0,
                          page_number=1, text=text, score=score)


class TestTokenizer:
    def test_lowercases_and_drops_stopwords(self):
        assert tokenize("The Quick Brown Fox") == ["quick", "brown", "fox"]

    def test_keeps_identifiers_intact(self):
        assert "nw-4417" in tokenize("Contract NW-4417 renews soon")

    def test_keeps_numbers(self):
        assert "2024" in tokenize("filed in 2024")


class TestReciprocalRankFusion:
    def test_agreement_between_retrievers_wins(self):
        fused = reciprocal_rank_fusion(
            {"dense": ["a", "b", "c"], "sparse": ["b", "a", "d"]}, k=60
        )
        assert fused["b"] > fused["c"]
        assert fused["a"] > fused["d"]

    def test_rank_one_beats_rank_two(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b"]}, k=60)
        assert fused["a"] > fused["b"]

    def test_documents_found_by_both_arms_outrank_singletons(self):
        fused = reciprocal_rank_fusion({"dense": ["x", "y"], "sparse": ["x", "z"]}, k=60)
        assert fused["x"] > fused["y"] and fused["x"] > fused["z"]

    def test_empty_input_is_safe(self):
        assert reciprocal_rank_fusion({}, k=60) == {}


class TestMaximalMarginalRelevance:
    def test_prefers_diversity_over_near_duplicates(self):
        query = [1.0, 0.0]
        candidates = [chunk("a"), chunk("b"), chunk("c")]
        vectors = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
        # lambda 0.5 is an exact tie here (b's relevance and its redundancy against
        # a are both ~1.0), so lean the weighting toward diversity to test it.
        picked = [c.chunk_id for c in
                  maximal_marginal_relevance(query, candidates, vectors, 2, lambda_=0.3)]
        assert picked[0] == "a" and picked[1] == "c"

    def test_lambda_one_is_pure_relevance(self):
        query = [1.0, 0.0]
        candidates = [chunk("a"), chunk("b"), chunk("c")]
        vectors = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
        picked = [c.chunk_id for c in
                  maximal_marginal_relevance(query, candidates, vectors, 2, lambda_=1.0)]
        assert picked == ["a", "b"]

    def test_single_candidate_passes_through(self):
        assert len(maximal_marginal_relevance([1.0], [chunk("a")], {"a": [1.0]}, 5, 0.5)) == 1


class TestHeuristicReranker:
    def test_scores_term_overlap_higher(self):
        candidates = [
            chunk("hit", text="Total revenue for Q3 reached 48.2 million dollars."),
            chunk("miss", text="The office cafeteria now serves breakfast."),
        ]
        scores = Reranker._heuristic_scores("What was Q3 revenue?", candidates)
        assert scores["hit"] > scores["miss"]

    def test_survives_a_stopword_only_query(self):
        candidates = [chunk("a", score=0.7)]
        assert Reranker._heuristic_scores("the and of", candidates) == {"a": 0.7}

    def test_rerank_falls_back_when_the_mode_is_off(self):
        candidates = [chunk("a", score=0.9), chunk("b", score=0.1)]
        result = Reranker().rerank("anything", candidates, top_k=1, mode="off")
        assert [c.chunk_id for c in result["chunks"]] == ["a"]

    def test_heuristic_mode_reorders_candidates(self):
        candidates = [
            chunk("miss", score=0.9, text="Unrelated cafeteria menu information."),
            chunk("hit", score=0.4, text="Gross margin was 74.9 percent this quarter."),
        ]
        result = Reranker().rerank("What was the gross margin?", candidates,
                                   top_k=2, mode="heuristic")
        assert result["chunks"][0].chunk_id == "hit"
        assert result["mode"] == "heuristic"

    def test_empty_candidates_are_safe(self):
        assert Reranker().rerank("q", [], top_k=5)["chunks"] == []
