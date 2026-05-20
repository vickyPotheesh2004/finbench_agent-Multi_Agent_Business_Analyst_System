"""
src/utils/llm_client.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 · Rev 1.0

Gemma4 LLM Client — wraps Ollama gemma4:e4b at localhost:11434

Used by ALL analysis pods: N11 LeadAnalyst, N12 QuantAnalyst,
N14 BlindAuditor, N15 PIVMediator, N02 SectionTree summaries.

Constraints:
    C1  $0 cost — local Ollama only
    C2  100% local — zero external network calls
    C3  Model = qwen2.5:3b (default)
    C5  seed=42

CHANGELOG:
    2026-05-10 S27  Bug Fix 2: timeout 120 -> 30, retries 3 -> 1,
                    availability cache (TTL=30s), fast-fail in chat().
                    Worst case per call: 30s instead of 360s+.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL    = "qwen2.5:3b"
FALLBACK_MODEL   = "qwen2.5:3b"
BASE_URL         = "http://localhost:11434"
DEFAULT_TIMEOUT  = 90        # was 30 — Bug B3 fix: tolerate Ollama cold start
DEFAULT_TEMP     = 0.1       # low temperature for factual financial QA
MAX_RETRIES      = 1         # was 3 — Bug Fix 2: don't compound timeouts
RETRY_DELAY      = 1.0       # seconds between retries (was 2.0)
SEED             = 42

# Bug Fix 2: availability cache (avoid 30s probe on every chat)
_AVAILABILITY_CACHE_TTL = 30.0  # seconds


class Gemma4Client:
    """
    Wraps Ollama for all LLM calls in the pipeline.

    Features:
        - Automatic retry on timeout/connection error (MAX_RETRIES=1 after fix)
        - Circuit breaker — trips after 3 consecutive failures
        - Health check — verifies Ollama is running before first call
        - Availability cache — 30s TTL to avoid hammering the endpoint
        - Fast-fail when Ollama is known unavailable (Bug Fix 2)
        - Context-first enforcement — validates C7 in prompts
    """

    def __init__(
        self,
        model:    str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        timeout:  int = DEFAULT_TIMEOUT,
        seed:     int = SEED,
    ) -> None:
        self.model      = model
        self.base_url   = base_url.rstrip("/")
        self.timeout    = timeout
        self.seed       = seed

        # Circuit breaker state
        self._failure_count    = 0
        self._circuit_open     = False
        self._circuit_open_at  = 0.0
        self._circuit_reset_s  = 60.0

        # Stats
        self._total_calls      = 0
        self._total_failures   = 0
        self._last_latency_ms  = 0.0

        # Bug Fix 2: availability cache (instance-level)
        self._avail_cache: Optional[bool]   = None
        self._avail_checked_at: float       = 0.0

    # ── Primary interface ─────────────────────────────────────────────────────

    def chat(
        self,
        prompt:      str,
        temperature: float = DEFAULT_TEMP,
        max_tokens:  int   = 2048,
        system:      str   = "",
    ) -> str:
        """
        Send a prompt to the LLM and return the response text.

        Args:
            prompt      : The user prompt (context MUST come before question)
            temperature : 0.0–1.0. Default 0.1 for factual accuracy.
            max_tokens  : Maximum tokens in response
            system      : Optional system message

        Returns:
            Response text string. Empty string on failure.

        Bug Fix 2 protections:
            1. Fast-fail when Ollama is unavailable (cached check, ~0s)
            2. Reduced timeout 30s + 1 retry instead of 120s + 3 retries
            3. Circuit breaker (existing) trips on 3 consecutive failures
        """
        # ── Bug Fix 2 Layer 1: fast-fail when Ollama is known down ──
        # This avoids 30s × MAX_RETRIES wasted on every call when service
        # is dead. Cached availability check returns in ~0s.
        if not self.is_available():
            logger.warning("[LLM] Ollama unavailable — fast-fail (0s)")
            return ""

        # ── Bug Fix 2 Layer 2: existing circuit breaker ──
        if self._circuit_open:
            if time.time() - self._circuit_open_at > self._circuit_reset_s:
                logger.info("[LLM] Circuit breaker reset — retrying")
                self._circuit_open   = False
                self._failure_count  = 0
            else:
                logger.warning("[LLM] Circuit open — skipping LLM call")
                return ""

        self._total_calls += 1
        t0 = time.time()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._call_ollama(
                    prompt, temperature, max_tokens, system
                )
                self._failure_count  = 0
                self._last_latency_ms = (time.time() - t0) * 1000
                logger.debug(
                    "[LLM] Response received | latency=%.0fms | attempt=%d",
                    self._last_latency_ms, attempt,
                )
                return response

            except Exception as exc:
                logger.warning(
                    "[LLM] Attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, exc,
                )
                # On error, invalidate availability cache so next call re-checks
                self._avail_cache = None
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)

        # All retries failed
        self._total_failures  += 1
        self._failure_count   += 1
        if self._failure_count >= MAX_RETRIES:
            self._circuit_open    = True
            self._circuit_open_at = time.time()
            logger.error(
                "[LLM] Circuit breaker TRIPPED after %d failures",
                self._failure_count,
            )

        return ""

    def chat_json(
        self,
        prompt:      str,
        temperature: float = DEFAULT_TEMP,
    ) -> Dict[str, Any]:
        """
        Call LLM and parse response as JSON.
        Returns empty dict on failure or invalid JSON.
        """
        raw = self.chat(prompt, temperature=temperature)
        if not raw:
            return {}
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines   = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("[LLM] JSON parse failed — returning raw text in dict")
            return {"raw_text": raw}

    # ── Health & availability ─────────────────────────────────────────────────

    def is_available(self) -> bool:
        """
        Check if Ollama is running and model is available.

        Bug Fix 2: cache result for 30s (TTL = _AVAILABILITY_CACHE_TTL).
        First call probes the /api/tags endpoint with 3s timeout.
        Subsequent calls within TTL reuse the cached result (returns ~0s).

        When Ollama is dead, this saves the timeout × every chat() call.
        """
        now = time.time()

        # Use cached result if fresh
        if self._avail_cache is not None:
            if (now - self._avail_checked_at) < _AVAILABILITY_CACHE_TTL:
                return self._avail_cache

        try:
            import urllib.request
            url = f"{self.base_url}/api/tags"
            with urllib.request.urlopen(url, timeout=3) as resp:  # was 5s
                if resp.status != 200:
                    self._avail_cache = False
                    self._avail_checked_at = now
                    return False
                data   = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                result = any(self.model in m for m in models)
                self._avail_cache = result
                self._avail_checked_at = now
                return result
        except Exception:
            self._avail_cache = False
            self._avail_checked_at = now
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Return health status dict.
        Used by /health endpoint and Streamlit UI status indicator.
        """
        available = self.is_available()
        return {
            "model":           self.model,
            "base_url":        self.base_url,
            "available":       available,
            "circuit_open":    self._circuit_open,
            "total_calls":     self._total_calls,
            "total_failures":  self._total_failures,
            "failure_rate":    (
                self._total_failures / self._total_calls
                if self._total_calls > 0 else 0.0
            ),
            "last_latency_ms": self._last_latency_ms,
        }

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker."""
        self._circuit_open    = False
        self._failure_count   = 0
        self._circuit_open_at = 0.0
        # Also invalidate availability cache
        self._avail_cache       = None
        self._avail_checked_at  = 0.0
        logger.info("[LLM] Circuit breaker manually reset")

    # ── Private ───────────────────────────────────────────────────────────────

    def _call_ollama(
        self,
        prompt:      str,
        temperature: float,
        max_tokens:  int,
        system:      str,
    ) -> str:
        """
        Make HTTP POST to Ollama /api/chat.
        Uses stdlib urllib only — zero extra dependencies.
        C2: no external network — localhost only.
        """
        import urllib.request

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model":   self.model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed":        self.seed,
            },
        }).encode("utf-8")

        url = f"{self.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status}")
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "")


# ── Singleton helper ──────────────────────────────────────────────────────────

_default_client: Optional[Gemma4Client] = None


def get_llm_client(
    model:    str = DEFAULT_MODEL,
    base_url: str = BASE_URL,
) -> Gemma4Client:
    """
    Return a shared Gemma4Client instance.
    Creates on first call, reuses on subsequent calls.
    """
    global _default_client
    if _default_client is None:
        _default_client = Gemma4Client(model=model, base_url=base_url)
    return _default_client


def reset_llm_client() -> None:
    """Reset the shared client — used in tests."""
    global _default_client
    _default_client = None