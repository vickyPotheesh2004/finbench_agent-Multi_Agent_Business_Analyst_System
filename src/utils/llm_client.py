"""
src/utils/llm_client.py

Production-Grade Local LLM Client
Optimized for:

- Ollama
- Llama 3.1
- Windows
- Colab
- T4 GPUs
- FinanceBench
- Long-running evaluation
- Stability under load
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

from typing import Any
from typing import Dict
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama3.1:8b"

FALLBACK_MODEL = "llama3.1:8b"

BASE_URL = "http://127.0.0.1:11434"

# Increased for FinanceBench
DEFAULT_TIMEOUT = 180

DEFAULT_TEMP = 0.1

MAX_RETRIES = 3

RETRY_DELAY = 2.0

SEED = 42

AVAILABILITY_CACHE_TTL = 30.0

CIRCUIT_RESET_SECONDS = 60.0

HEALTH_TIMEOUT = 5

MAX_PROMPT_CHARS = 120_000

MAX_RESPONSE_CHARS = 40_000

KEEP_ALIVE = "1h"

# ─────────────────────────────────────────────────────────────────────────────
# Ollama Client
# ─────────────────────────────────────────────────────────────────────────────


class Gemma4Client:

    """
    Local Ollama client.

    Features:
    - Thread-safe
    - Retry hardened
    - Circuit breaker
    - Long-context safe
    - JSON-safe
    - Deterministic inference
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        seed: int = SEED,
    ) -> None:

        self.model = (
            model.strip()
        )

        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = max(
            30,
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

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

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

        last_error = None

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

                last_error = exc

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

        logger.error(
            "[LLM] Failed after retries: %s",
            last_error,
        )

        return ""

    # ─────────────────────────────────────────────────────────────────────
    # JSON API
    # ─────────────────────────────────────────────────────────────────────

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

        cleaned = (
            cleaned.strip()
        )

        # Try direct parse

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

        except Exception:
            pass

        # Try extraction parse

        try:

            start = cleaned.find("{")

            end = cleaned.rfind("}")

            if (
                start >= 0
                and end > start
            ):

                extracted = cleaned[
                    start:end + 1
                ]

                parsed = json.loads(
                    extracted
                )

                if isinstance(
                    parsed,
                    dict,
                ):
                    return parsed

        except Exception:
            pass

        logger.debug(
            "[LLM] Invalid JSON response"
        )

        return {
            "raw_text": raw
        }

    # ─────────────────────────────────────────────────────────────────────
    # Availability
    # ─────────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────────
    # Health
    # ─────────────────────────────────────────────────────────────────────

    def health_check(
        self,
    ) -> Dict[str, Any]:

        total_calls = max(
            1,
            self._total_calls,
        )

        return {
            "model": self.model,
            "available": self.is_available(),
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

    # ─────────────────────────────────────────────────────────────────────
    # Ollama HTTP
    # ─────────────────────────────────────────────────────────────────────

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

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": KEEP_ALIVE,
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

        request = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(
                payload
            ).encode("utf-8"),
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

        except Exception as exc:

            raise RuntimeError(
                f"Ollama request failed: {exc}"
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

        if not isinstance(
            content,
            str,
        ):
            content = str(content)

        if (
            len(content)
            > MAX_RESPONSE_CHARS
        ):

            content = content[
                :MAX_RESPONSE_CHARS
            ]

        return content.strip()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

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
# Public API
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