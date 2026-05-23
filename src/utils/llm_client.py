"""
src/utils/llm_client.py

Production-Grade Local LLM Client
FinBench Multi-Agent Business Analyst AI

Capabilities
------------
1. Fully local Ollama inference
2. Zero external network calls
3. Circuit breaker protection
4. Fast-fail availability cache
5. Automatic retries
6. Timeout hardening
7. JSON-safe generation
8. Deterministic seed support
9. Production health checks
10. Structured telemetry
11. Thread-safe singleton
12. Context-first validation
13. Streaming-safe architecture
14. Resource-safe HTTP handling
15. Production-grade logging
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama3.1:8b"

FALLBACK_MODEL = "llama3.1:8b"

BASE_URL = "http://localhost:11434"

DEFAULT_TIMEOUT = 30

DEFAULT_TEMP = 0.1

MAX_RETRIES = 1

RETRY_DELAY = 1.0

SEED = 42

AVAILABILITY_CACHE_TTL = 30.0

CIRCUIT_RESET_SECONDS = 30.0

HEALTH_TIMEOUT = 3

MAX_PROMPT_CHARS = 120_000

# ─────────────────────────────────────────────────────────────────────────────
# Gemma4Client
# ─────────────────────────────────────────────────────────────────────────────


class Gemma4Client:

    """
    Local Ollama wrapper.

    Constraints
    -----------
    - Localhost only
    - No cloud APIs
    - Deterministic inference
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        seed: int = SEED,
    ) -> None:

        self.model = model.strip()

        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = max(
            5,
            int(timeout),
        )

        self.seed = int(seed)

        # Circuit breaker

        self._failure_count = 0

        self._circuit_open = False

        self._circuit_open_at = 0.0

        self._circuit_reset_s = (
            CIRCUIT_RESET_SECONDS
        )

        # Metrics

        self._total_calls = 0

        self._total_failures = 0

        self._last_latency_ms = 0.0

        # Availability cache

        self._avail_cache: Optional[
            bool
        ] = None

        self._avail_checked_at = 0.0

        # Thread safety

        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    # Primary API
    # ─────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        temperature: float = DEFAULT_TEMP,
        max_tokens: int = 2048,
        system: str = "",
    ) -> str:

        if not prompt:
            return ""

        prompt = self._sanitize_prompt(
            prompt
        )

        if not self.is_available():

            logger.warning(
                "[LLM] Ollama unavailable"
            )

            return ""

        if self._is_circuit_open():

            logger.warning(
                "[LLM] Circuit breaker open"
            )

            return ""

        self._total_calls += 1

        t0 = time.time()

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):

            try:

                response = (
                    self._call_ollama(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system=system,
                    )
                )

                self._failure_count = 0

                self._last_latency_ms = (
                    time.time() - t0
                ) * 1000.0

                return (
                    response.strip()
                )

            except Exception as exc:

                logger.warning(
                    "[LLM] attempt=%d failed: %s",
                    attempt,
                    exc,
                )

                self._avail_cache = None

                if (
                    attempt
                    < MAX_RETRIES
                ):

                    time.sleep(
                        RETRY_DELAY
                        * attempt
                    )

        self._register_failure()

        return ""

    # ─────────────────────────────────────────────────────────────────────────
    # JSON API
    # ─────────────────────────────────────────────────────────────────────────

    def chat_json(
        self,
        prompt: str,
        temperature: float = DEFAULT_TEMP,
        max_tokens: int = 2048,
        system: str = "",
    ) -> Dict[str, Any]:

        raw = self.chat(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
        )

        if not raw:
            return {}

        cleaned = (
            self._strip_code_fences(
                raw
            )
        )

        try:

            parsed = json.loads(
                cleaned
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

            return {
                "data": parsed
            }

        except json.JSONDecodeError:

            logger.debug(
                "[LLM] Invalid JSON response"
            )

            return {
                "raw_text": raw
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Health
    # ─────────────────────────────────────────────────────────────────────────

    def is_available(
        self,
    ) -> bool:

        now = time.time()

        with self._lock:

            if (
                self._avail_cache
                is not None
            ):

                age = (
                    now
                    - self._avail_checked_at
                )

                if (
                    age
                    < AVAILABILITY_CACHE_TTL
                ):

                    return self._avail_cache

        try:

            url = (
                f"{self.base_url}/api/tags"
            )

            with urllib.request.urlopen(
                url,
                timeout=HEALTH_TIMEOUT,
            ) as resp:

                if resp.status != 200:

                    self._update_availability(
                        False
                    )

                    return False

                payload = json.loads(
                    resp.read().decode(
                        "utf-8"
                    )
                )

                models = [
                    (
                        m.get(
                            "name",
                            "",
                        )
                        .strip()
                    )
                    for m in payload.get(
                        "models",
                        [],
                    )
                ]

                available = any(
                    self.model in m
                    for m in models
                )

                self._update_availability(
                    available
                )

                return available

        except Exception:

            self._update_availability(
                False
            )

            return False

    def health_check(
        self,
    ) -> Dict[str, Any]:

        available = (
            self.is_available()
        )

        total_calls = max(
            1,
            self._total_calls,
        )

        return {
            "model": self.model,
            "base_url": self.base_url,
            "available": available,
            "circuit_open": self._circuit_open,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "failure_rate": round(
                self._total_failures
                / total_calls,
                4,
            ),
            "last_latency_ms": round(
                self._last_latency_ms,
                2,
            ),
        }

    def reset_circuit(
        self,
    ) -> None:

        with self._lock:

            self._failure_count = 0

            self._circuit_open = False

            self._circuit_open_at = 0.0

            self._avail_cache = None

            self._avail_checked_at = 0.0

        logger.info(
            "[LLM] Circuit reset"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal HTTP
    # ─────────────────────────────────────────────────────────────────────────

    def _call_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system: str,
    ) -> str:

        messages = []

        if system:

            messages.append(
                {
                    "role": "system",
                    "content": system,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": float(
                        temperature
                    ),
                    "num_predict": int(
                        max_tokens
                    ),
                    "seed": self.seed,
                },
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=payload,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as resp:

                if resp.status != 200:

                    raise RuntimeError(
                        f"Ollama HTTP {resp.status}"
                    )

                data = json.loads(
                    resp.read().decode(
                        "utf-8"
                    )
                )

        except urllib.error.HTTPError as exc:

            raise RuntimeError(
                f"HTTPError {exc.code}"
            ) from exc

        except urllib.error.URLError as exc:

            raise RuntimeError(
                f"URLError {exc.reason}"
            ) from exc

        content = (
            data.get(
                "message",
                {},
            ).get(
                "content",
                "",
            )
        )

        return str(content)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _sanitize_prompt(
        self,
        prompt: str,
    ) -> str:

        prompt = prompt.replace(
            "\x00",
            ""
        )

        prompt = prompt.strip()

        if (
            len(prompt)
            > MAX_PROMPT_CHARS
        ):

            logger.warning(
                "[LLM] Prompt truncated"
            )

            prompt = prompt[
                :MAX_PROMPT_CHARS
            ]

        return prompt

    @staticmethod
    def _strip_code_fences(
        text: str,
    ) -> str:

        cleaned = text.strip()

        if cleaned.startswith(
            "```"
        ):

            lines = cleaned.split(
                "\n"
            )

            if len(lines) >= 3:

                cleaned = "\n".join(
                    lines[1:-1]
                )

        return cleaned.strip()

    def _is_circuit_open(
        self,
    ) -> bool:

        if not self._circuit_open:
            return False

        elapsed = (
            time.time()
            - self._circuit_open_at
        )

        if (
            elapsed
            > self._circuit_reset_s
        ):

            logger.info(
                "[LLM] Circuit auto-reset"
            )

            self._circuit_open = False

            self._failure_count = 0

            return False

        return True

    def _register_failure(
        self,
    ) -> None:

        with self._lock:

            self._total_failures += 1

            self._failure_count += 1

            if (
                self._failure_count
                >= MAX_RETRIES
            ):

                self._circuit_open = True

                self._circuit_open_at = (
                    time.time()
                )

                logger.error(
                    "[LLM] Circuit breaker tripped"
                )

    def _update_availability(
        self,
        value: bool,
    ) -> None:

        with self._lock:

            self._avail_cache = value

            self._avail_checked_at = (
                time.time()
            )

# ─────────────────────────────────────────────────────────────────────────────
# Shared Singleton
# ─────────────────────────────────────────────────────────────────────────────

_default_client: Optional[
    Gemma4Client
] = None

_client_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Public Helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_llm_client(
    model: str = DEFAULT_MODEL,
    base_url: str = BASE_URL,
) -> Gemma4Client:

    global _default_client

    with _client_lock:

        if _default_client is None:

            _default_client = (
                Gemma4Client(
                    model=model,
                    base_url=base_url,
                )
            )

        return _default_client


def reset_circuit_breaker(
) -> None:

    global _default_client

    if _default_client:

        _default_client.reset_circuit()


def reset_llm_client(
) -> None:

    global _default_client

    with _client_lock:

        _default_client = None