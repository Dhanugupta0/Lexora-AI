"""Embedder: role handling, caching and provider fallback.

Nothing here touches the network. The Jina provider is exercised through its
request payload and its response parsing, which is where the bugs that silently
cost recall actually live.
"""

import pytest

from app.rag.embedder import ROLES, Embedder, _JinaProvider


class _StubProvider:
    """Records every call so we can assert on batching, roles and caching."""

    name = "stub"
    model_name = "stub-model"
    dimension = 4

    def __init__(self):
        self.calls = []

    def encode(self, texts, role="passage"):
        self.calls.append((list(texts), role))
        return [[float(len(t)), 0.0, 0.0, 1.0] for t in texts]


def embedder_with(provider):
    embedder = Embedder()
    embedder._provider = provider          # skip the real provider chain
    return embedder


class TestRoles:
    def test_query_and_passage_reach_the_provider_with_their_role(self):
        stub = _StubProvider()
        embedder = embedder_with(stub)

        embedder.embed_query("what is the refund window?")
        embedder.embed_documents(["Refunds are issued within 30 days."])
        embedder.embed_similarity(["a claim"])

        assert [role for _, role in stub.calls] == ["query", "passage", "similarity"]

    def test_unknown_role_falls_back_to_passage(self):
        stub = _StubProvider()
        embedder_with(stub).embed(["text"], "nonsense")
        assert stub.calls[0][1] == "passage"

    def test_every_declared_role_is_accepted_unchanged(self):
        stub = _StubProvider()
        embedder = embedder_with(stub)
        for role in ROLES:
            embedder.embed([f"text for {role}"], role)
        assert [role for _, role in stub.calls] == list(ROLES)


class TestCache:
    def test_repeated_text_is_embedded_once(self):
        stub = _StubProvider()
        embedder = embedder_with(stub)

        first = embedder.embed(["same text"], "passage")
        second = embedder.embed(["same text"], "passage")

        assert first == second
        assert len(stub.calls) == 1

    def test_same_text_in_two_roles_is_not_a_cache_hit(self):
        """A question and a passage that happen to share wording are different
        vectors -- collapsing them would silently break asymmetric retrieval."""
        stub = _StubProvider()
        embedder = embedder_with(stub)

        embedder.embed(["identical"], "query")
        embedder.embed(["identical"], "passage")

        assert len(stub.calls) == 2

    def test_partial_hits_only_send_the_missing_texts(self):
        stub = _StubProvider()
        embedder = embedder_with(stub)

        embedder.embed(["a"], "passage")
        embedder.embed(["a", "bb"], "passage")

        assert stub.calls[1][0] == ["bb"]

    def test_results_keep_input_order_when_mixing_hits_and_misses(self):
        stub = _StubProvider()
        embedder = embedder_with(stub)

        embedder.embed(["bb"], "passage")
        vectors = embedder.embed(["a", "bb", "ccc"], "passage")

        # the stub encodes length into the first component
        assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


class TestResilience:
    def test_a_failing_batch_is_bisected_to_isolate_the_bad_input(self):
        class Flaky(_StubProvider):
            def encode(self, texts, role="passage"):
                if "poison" in texts:
                    raise RuntimeError("provider rejected the input")
                return super().encode(texts, role)

        vectors = embedder_with(Flaky()).embed(["ok", "poison", "fine"], "passage")

        assert len(vectors) == 3
        assert vectors[1] == [0.0] * 4          # only the bad one degrades
        assert vectors[0][0] == 2.0 and vectors[2][0] == 4.0

    def test_empty_input_never_calls_the_provider(self):
        stub = _StubProvider()
        assert embedder_with(stub).embed([]) == []
        assert stub.calls == []


class TestJinaPayload:
    """`_payload` is pure, so we can build it without constructing the client."""

    def payload(self, texts, role, model="jina-embeddings-v4", dimensions=1024):
        provider = _JinaProvider.__new__(_JinaProvider)     # no network, no key
        provider.model_name = model
        provider._dimensions = dimensions
        return provider._payload(texts, role)

    @pytest.mark.parametrize("role,task", [
        ("query", "retrieval.query"),
        ("passage", "retrieval.passage"),
        ("similarity", "text-matching"),
    ])
    def test_roles_map_onto_the_models_typed_tasks(self, role, task):
        assert self.payload(["x"], role)["task"] == task

    def test_unknown_role_defaults_to_the_passage_task(self):
        assert self.payload(["x"], "bogus")["task"] == "retrieval.passage"

    def test_task_values_are_the_ones_the_api_accepts(self):
        allowed = {"text-matching", "retrieval.query", "retrieval.passage",
                   "code.query", "code.passage"}
        assert set(_JinaProvider._TASKS.values()) <= allowed

    def test_input_uses_the_multimodal_object_form(self):
        assert self.payload(["one", "two"], "passage")["input"] == [
            {"text": "one"}, {"text": "two"}
        ]

    def test_dimensions_are_sent_when_configured(self):
        assert self.payload(["x"], "passage", dimensions=1024)["dimensions"] == 1024

    def test_dimensions_are_omitted_when_unset_so_the_model_default_applies(self):
        assert "dimensions" not in self.payload(["x"], "passage", dimensions=None)

    def test_long_input_is_truncated_rather_than_rejected(self):
        assert self.payload(["x"], "passage")["truncate"] is True

    def test_model_name_is_forwarded(self):
        assert self.payload(["x"], "passage")["model"] == "jina-embeddings-v4"
