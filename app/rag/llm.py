"""Groq chat client with retries, a model fallback chain and a circuit breaker.

Everything that talks to an LLM in this project goes through `get_llm()`. That
gives us one place to enforce timeouts, back off on rate limits, fail over to a
different model, and stop hammering a provider that is clearly down.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMUnavailable(RuntimeError):
    """Every model in the chain failed, or the circuit is open."""


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int
    attempts: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    degraded: bool = False          # True when a fallback model produced this


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #
class CircuitBreaker:
    """Classic three-state breaker: closed -> open -> half-open -> closed.

    After `threshold` consecutive failures the circuit opens and calls fail fast
    for `cooldown` seconds. The first call after the cooldown is a probe; if it
    succeeds the breaker closes again.
    """

    def __init__(self, threshold: int, cooldown: float):
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._failures < self.threshold:
                return "closed"
            if time.time() - self._opened_at >= self.cooldown:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._opened_at = time.time()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "consecutive_failures": self._failures,
            "threshold": self.threshold,
        }


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
_TRANSIENT_MARKERS = (
    "rate limit", "rate_limit", "429", "timeout", "timed out", "temporarily",
    "503", "502", "500", "overloaded", "connection", "unavailable",
)


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 425, 429, 500, 502, 503, 504):
        return True
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _TRANSIENT_MARKERS)


_RETRY_AFTER_TEXT = re.compile(r"try again in ([\d.]+)\s*(ms|s)\b", re.I)


def _retry_after(exc: Exception) -> Optional[float]:
    """Groq tells us exactly how long to wait -- back off for at least that long.

    Guessing shorter than the server's own hint just burns a retry attempt, which
    is what pushes a request onto the fallback model unnecessarily.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        for key in ("retry-after", "retry-after-ms", "x-ratelimit-reset-tokens"):
            raw = headers.get(key)
            if not raw:
                continue
            try:
                value = float(str(raw).rstrip("smh"))
            except ValueError:
                continue
            return value / 1000.0 if key.endswith("-ms") else value

    match = _RETRY_AFTER_TEXT.search(str(exc))
    if match:
        value = float(match.group(1))
        return value / 1000.0 if match.group(2).lower() == "ms" else value
    return None


def _is_fatal_for_model(exc: Exception) -> bool:
    """Errors where retrying the *same* model is pointless -- move to the next."""
    status = getattr(exc, "status_code", None)
    if status in (400, 404, 413, 422):
        return True
    blob = str(exc).lower()
    return "does not exist" in blob or "decommissioned" in blob or "context" in blob and "length" in blob


class LLMClient:
    def __init__(self) -> None:
        self._client = None
        self._lock = threading.Lock()
        self.breaker = CircuitBreaker(
            settings.CIRCUIT_FAIL_THRESHOLD, settings.CIRCUIT_COOLDOWN_SECONDS
        )
        self.models = settings.model_chain
        self._stats = {"calls": 0, "failures": 0, "fallbacks": 0, "retries": 0}

    # -- lazy client ------------------------------------------------------- #
    def _groq(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    if not settings.GROQ_API_KEY:
                        raise LLMUnavailable("GROQ_API_KEY is not set.")
                    from groq import Groq
                    self._client = Groq(
                        api_key=settings.GROQ_API_KEY,
                        timeout=settings.LLM_TIMEOUT_SECONDS,
                        max_retries=0,          # we own the retry policy
                    )
        return self._client

    @property
    def available(self) -> bool:
        return bool(settings.GROQ_API_KEY) and self.breaker.allow()

    # -- core call --------------------------------------------------------- #
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        tag: str = "generic",
    ) -> LLMResult:
        """Run a chat completion, retrying and failing over as needed."""
        if not settings.GROQ_API_KEY:
            raise LLMUnavailable("GROQ_API_KEY is not set.")
        if not self.breaker.allow():
            raise LLMUnavailable(
                f"LLM circuit is open (cooling down for "
                f"{settings.CIRCUIT_COOLDOWN_SECONDS:.0f}s after repeated failures)."
            )

        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE if temperature is None else temperature,
            "max_completion_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.time()
        attempts = 0
        last_error: Optional[Exception] = None

        for model_index, model in enumerate(self.models):
            call = dict(payload, model=model)
            # `reasoning_effort` is a gpt-oss feature; other models reject it.
            if "gpt-oss" in model:
                call["reasoning_effort"] = reasoning_effort or settings.LLM_REASONING_EFFORT

            for attempt in range(settings.LLM_MAX_RETRIES):
                attempts += 1
                try:
                    response = self._groq().chat.completions.create(**call)
                    text = (response.choices[0].message.content or "").strip()
                    if not text:
                        raise RuntimeError("model returned an empty completion")

                    self.breaker.record_success()
                    self._stats["calls"] += 1
                    if model_index > 0:
                        self._stats["fallbacks"] += 1
                    usage = getattr(response, "usage", None)
                    return LLMResult(
                        text=text,
                        model=model,
                        latency_ms=int((time.time() - started) * 1000),
                        attempts=attempts,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        degraded=model_index > 0,
                    )

                except Exception as exc:                       # noqa: BLE001
                    last_error = exc
                    self._stats["failures"] += 1
                    if _is_fatal_for_model(exc):
                        logger.warning("[llm:%s] %s unusable (%s) -- next model", tag, model, exc)
                        break
                    if attempt < settings.LLM_MAX_RETRIES - 1 and _is_transient(exc):
                        delay = settings.LLM_BACKOFF_BASE * (2 ** attempt)
                        delay += random.uniform(0, delay * 0.3)   # jitter
                        hinted = _retry_after(exc)
                        if hinted is not None:
                            # Respect the server's hint, but never stall the request.
                            delay = min(max(delay, hinted + 0.25), settings.LLM_TIMEOUT_SECONDS / 2)
                        self._stats["retries"] += 1
                        logger.warning(
                            "[llm:%s] %s attempt %d failed (%s) -- retrying in %.1fs",
                            tag, model, attempt + 1, exc, delay,
                        )
                        time.sleep(delay)
                        continue
                    logger.warning("[llm:%s] %s failed (%s) -- next model", tag, model, exc)
                    break

        self.breaker.record_failure()
        raise LLMUnavailable(f"All models failed for '{tag}'. Last error: {last_error}")

    # -- json convenience -------------------------------------------------- #
    def complete_json(
        self,
        messages: List[Dict[str, str]],
        *,
        default: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Like `complete`, but parses JSON and never raises on bad output.

        Returns `default` (or {}) when the call fails or the model emits
        something unparseable -- callers always get a dict.
        """
        try:
            result = self.complete(messages, json_mode=True, **kwargs)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("[llm:%s] json call failed: %s", kwargs.get("tag", "json"), exc)
            return dict(default or {})

        parsed = extract_json(result.text)
        if parsed is None:
            logger.warning("[llm:%s] unparseable JSON: %r", kwargs.get("tag", "json"), result.text[:200])
            return dict(default or {})
        parsed.setdefault("_model", result.model)
        parsed.setdefault("_latency_ms", result.latency_ms)
        return parsed

    def snapshot(self) -> Dict[str, Any]:
        return {
            "provider": "groq",
            "model_chain": self.models,
            "configured": bool(settings.GROQ_API_KEY),
            "circuit": self.breaker.snapshot(),
            "stats": dict(self._stats),
        }


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Tolerant JSON extraction -- handles fences and chatty prefixes."""
    if not text:
        return None
    candidates = [text.strip()]
    fence = _FENCE.search(text)
    if fence:
        candidates.insert(0, fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            continue
    return None


_llm: Optional[LLMClient] = None
_llm_lock = threading.Lock()


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = LLMClient()
    return _llm
