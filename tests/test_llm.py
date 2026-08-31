import time
from unittest.mock import MagicMock, patch

import pytest

from app.rag.llm import CircuitBreaker, LLMClient, LLMUnavailable, extract_json


class TestCircuitBreaker:
    def test_starts_closed(self):
        assert CircuitBreaker(3, 1.0).state == "closed" and CircuitBreaker(3, 1.0).allow()

    def test_opens_after_the_failure_threshold(self):
        breaker = CircuitBreaker(3, 60.0)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == "open" and not breaker.allow()

    def test_success_resets_the_counter(self):
        breaker = CircuitBreaker(3, 60.0)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        assert breaker.state == "closed"

    def test_half_opens_after_the_cooldown(self):
        breaker = CircuitBreaker(1, 0.05)
        breaker.record_failure()
        assert breaker.state == "open"
        time.sleep(0.08)
        assert breaker.state == "half_open" and breaker.allow()


class TestJsonExtraction:
    def test_parses_bare_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_parses_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parses_json_with_a_chatty_prefix(self):
        assert extract_json('Sure, here you go: {"a": 1}') == {"a": 1}

    def test_returns_none_for_garbage(self):
        assert extract_json("not json at all") is None

    def test_returns_none_for_empty_input(self):
        assert extract_json("") is None


def _response(text, model="m"):
    message = MagicMock()
    message.content = text
    choice = MagicMock(message=message)
    return MagicMock(choices=[choice], usage=MagicMock(prompt_tokens=1, completion_tokens=2))


class TestResilience:
    def test_missing_key_raises_immediately(self):
        client = LLMClient()
        with patch("app.rag.llm.settings.GROQ_API_KEY", ""):
            with pytest.raises(LLMUnavailable):
                client.complete([{"role": "user", "content": "hi"}])

    def test_retries_a_transient_error_then_succeeds(self):
        client = LLMClient()
        client.models = ["model-a"]
        groq = MagicMock()
        groq.chat.completions.create.side_effect = [
            RuntimeError("rate limit reached"), _response("done"),
        ]
        with patch.object(client, "_groq", return_value=groq), \
             patch("app.rag.llm.settings.GROQ_API_KEY", "k"), \
             patch("app.rag.llm.settings.LLM_BACKOFF_BASE", 0.0):
            result = client.complete([{"role": "user", "content": "hi"}])
        assert result.text == "done" and result.attempts == 2

    def test_falls_over_to_the_next_model(self):
        client = LLMClient()
        client.models = ["broken", "working"]
        groq = MagicMock()
        error = RuntimeError("model does not exist")
        groq.chat.completions.create.side_effect = [error, _response("recovered")]
        with patch.object(client, "_groq", return_value=groq), \
             patch("app.rag.llm.settings.GROQ_API_KEY", "k"):
            result = client.complete([{"role": "user", "content": "hi"}])
        assert result.text == "recovered"
        assert result.model == "working" and result.degraded

    def test_raises_once_every_model_is_exhausted(self):
        client = LLMClient()
        client.models = ["a", "b"]
        groq = MagicMock()
        groq.chat.completions.create.side_effect = RuntimeError("model does not exist")
        with patch.object(client, "_groq", return_value=groq), \
             patch("app.rag.llm.settings.GROQ_API_KEY", "k"):
            with pytest.raises(LLMUnavailable):
                client.complete([{"role": "user", "content": "hi"}])

    def test_an_open_circuit_fails_fast(self):
        client = LLMClient()
        for _ in range(client.breaker.threshold):
            client.breaker.record_failure()
        with patch("app.rag.llm.settings.GROQ_API_KEY", "k"):
            with pytest.raises(LLMUnavailable, match="circuit is open"):
                client.complete([{"role": "user", "content": "hi"}])

    def test_complete_json_returns_the_default_instead_of_raising(self):
        client = LLMClient()
        with patch.object(client, "complete", side_effect=LLMUnavailable("down")):
            assert client.complete_json([], default={"fallback": True}) == {"fallback": True}

    def test_complete_json_survives_unparseable_output(self):
        client = LLMClient()
        with patch.object(client, "complete", return_value=MagicMock(text="nonsense", model="m",
                                                                    latency_ms=1)):
            assert client.complete_json([], default={}) == {}


class TestRetryAfterHint:
    def test_parses_the_hint_from_the_error_message(self):
        from app.rag.llm import _retry_after
        assert _retry_after(RuntimeError("Rate limit reached. Please try again in 3.66s")) == 3.66

    def test_parses_a_millisecond_hint(self):
        from app.rag.llm import _retry_after
        assert _retry_after(RuntimeError("try again in 450ms")) == 0.45

    def test_prefers_the_retry_after_header(self):
        from app.rag.llm import _retry_after
        exc = RuntimeError("try again in 9s")
        exc.response = MagicMock(headers={"retry-after": "2"})
        assert _retry_after(exc) == 2.0

    def test_returns_none_without_a_hint(self):
        from app.rag.llm import _retry_after
        assert _retry_after(RuntimeError("something else broke")) is None
