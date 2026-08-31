from unittest.mock import patch

from app.rag.reasoning import plan_query


class TestFastPaths:
    def test_greetings_skip_retrieval(self):
        plan = plan_query("hey there!")
        assert plan.intent == "chitchat" and not plan.needs_retrieval
        assert plan.planner == "heuristic"

    def test_self_introductions_are_small_talk(self):
        assert plan_query("I am Dhanu").intent == "chitchat"

    def test_thanks_is_small_talk(self):
        assert plan_query("thanks!").intent == "chitchat"

    def test_library_questions_route_to_meta(self):
        plan = plan_query("what documents do I have?")
        assert plan.intent == "meta" and not plan.needs_retrieval

    def test_a_real_question_is_not_fast_pathed(self):
        with patch("app.rag.reasoning.settings.ENABLE_QUERY_PLANNING", False):
            plan = plan_query("What was revenue in the third quarter?")
        assert plan.intent == "document_qa" and plan.needs_retrieval


class TestPlannerFallback:
    def test_falls_back_to_the_raw_question_when_the_llm_is_down(self):
        with patch("app.rag.reasoning.get_llm") as get_llm:
            get_llm.return_value.available = False
            plan = plan_query("What is the gross margin?")
        assert plan.planner == "fallback"
        assert plan.standalone_question == "What is the gross margin?"
        assert plan.search_queries == ["What is the gross margin?"]

    def test_an_unparseable_plan_degrades_gracefully(self):
        with patch("app.rag.reasoning.get_llm") as get_llm:
            get_llm.return_value.available = True
            get_llm.return_value.complete_json.return_value = {}
            plan = plan_query("What is the gross margin?")
        assert plan.planner == "fallback" and plan.needs_retrieval

    def test_uses_the_llm_plan_when_it_is_valid(self):
        with patch("app.rag.reasoning.get_llm") as get_llm:
            get_llm.return_value.available = True
            get_llm.return_value.complete_json.return_value = {
                "intent": "document_qa",
                "standalone_question": "When is contract NW-4417 due?",
                "search_queries": ["contract NW-4417 renewal date"],
                "hypothetical_answer": "Contract NW-4417 is due in Q1 FY2025.",
                "needs_retrieval": True,
            }
            plan = plan_query("and when is it due?",
                              history=[{"role": "user", "content": "What is contract NW-4417?"}])
        assert plan.planner == "llm"
        assert plan.standalone_question == "When is contract NW-4417 due?"

    def test_hyde_probe_joins_the_retrieval_queries(self):
        with patch("app.rag.reasoning.get_llm") as get_llm, \
             patch("app.rag.reasoning.settings.ENABLE_HYDE", True):
            get_llm.return_value.available = True
            get_llm.return_value.complete_json.return_value = {
                "intent": "document_qa",
                "standalone_question": "What is the margin?",
                "search_queries": ["gross margin percentage"],
                "hypothetical_answer": "Gross margin was 74.9%.",
                "needs_retrieval": True,
            }
            plan = plan_query("What is the margin?")
        assert "Gross margin was 74.9%." in plan.retrieval_queries

    def test_an_unknown_intent_is_coerced_to_document_qa(self):
        with patch("app.rag.reasoning.get_llm") as get_llm:
            get_llm.return_value.available = True
            get_llm.return_value.complete_json.return_value = {
                "intent": "banana", "standalone_question": "q", "search_queries": ["q"],
            }
            assert plan_query("some question here").intent == "document_qa"


class TestRetrievalQuerySafetyNet:
    def test_the_users_own_wording_is_always_searched(self):
        from app.rag.reasoning import QueryPlan
        # A rewrite that wrongly dragged the old topic in.
        plan = QueryPlan(original_question="what about expenses?",
                         standalone_question="What is the expense policy for STD-441?",
                         search_queries=["STD-441 expense policy"])
        assert "what about expenses?" in plan.retrieval_queries

    def test_a_pure_follow_up_adds_no_verbatim_probe(self):
        from app.rag.reasoning import QueryPlan
        plan = QueryPlan(original_question="and what about that one?",
                         standalone_question="What is the remote work policy?",
                         search_queries=["remote work policy"])
        assert "and what about that one?" not in plan.retrieval_queries

    def test_duplicates_are_not_repeated(self):
        from app.rag.reasoning import QueryPlan
        plan = QueryPlan(original_question="What is the margin?",
                         standalone_question="What is the margin?",
                         search_queries=["What is the margin?", "gross margin"])
        assert plan.retrieval_queries.count("What is the margin?") == 0


class TestResolvedQuestion:
    def _plan(self, original, standalone):
        from app.rag.reasoning import QueryPlan
        return QueryPlan(original_question=original, standalone_question=standalone,
                         search_queries=[standalone], planner="llm")

    def test_a_pronoun_follow_up_uses_the_rewrite(self):
        plan = self._plan("and when is it due?", "When is contract NW-4417 due?")
        assert plan.resolved_question == "When is contract NW-4417 due?"

    def test_a_topic_switch_keeps_the_users_wording(self):
        # The rewrite wrongly fused the previous subject back in.
        plan = self._plan("what about expenses?", "What is the expense policy under STD-441?")
        assert plan.resolved_question == "what about expenses?"

    def test_a_bare_fragment_uses_the_rewrite(self):
        plan = self._plan("the second one?", "What is the second risk listed?")
        assert plan.resolved_question == "What is the second risk listed?"

    def test_a_self_contained_question_is_untouched(self):
        plan = self._plan("What is the gross margin?", "What is the gross margin?")
        assert plan.resolved_question == "What is the gross margin?"

    def test_a_heuristic_plan_always_uses_the_standalone(self):
        from app.rag.reasoning import QueryPlan
        plan = QueryPlan(original_question="hi", standalone_question="hi", planner="heuristic")
        assert plan.resolved_question == "hi"

    def test_both_phrasings_are_searched(self):
        plan = self._plan("what about expenses?", "What is the expense policy under STD-441?")
        assert "What is the expense policy under STD-441?" in plan.retrieval_queries


class TestAnaphoraDetection:
    def test_detects_a_pronoun(self):
        from app.rag.reasoning import needs_history_resolution
        assert needs_history_resolution("and when is it due?")

    def test_detects_a_bare_fragment(self):
        from app.rag.reasoning import needs_history_resolution
        assert needs_history_resolution("what about the second one?")

    def test_a_self_contained_question_needs_nothing(self):
        from app.rag.reasoning import needs_history_resolution
        assert not needs_history_resolution("What is the remote work policy?")

    def test_an_empty_question_needs_nothing(self):
        from app.rag.reasoning import needs_history_resolution
        assert not needs_history_resolution("")
