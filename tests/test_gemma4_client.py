"""
tests/test_gemma4_client.py
Tests for Gemma4 LLM Client
PDR-BAAAI-001 Rev 1.0
25 tests - no Ollama needed (mocked)
"""
import json
import time
import pytest
from unittest.mock import patch, MagicMock
from src.utils.llm_client import (
    Gemma4Client,
    get_llm_client,
    reset_llm_client,
    DEFAULT_MODEL,
    BASE_URL,
    DEFAULT_TEMP,
    MAX_RETRIES,
    SEED,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_llm_client()
    yield
    reset_llm_client()


@pytest.fixture
def client():
    return Gemma4Client()


def _mock_ollama(response_text: str):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__  = MagicMock(return_value=False)
    mock_resp.status    = 200
    mock_resp.read.return_value = json.dumps({
        "message": {"content": response_text}
    }).encode("utf-8")
    return mock_resp


class TestConstants:

    def test_01_default_model_is_llama_3_1(self):
        # 2026-06-05: DEFAULT_MODEL is llama3.1:8b for C3 compliance,
        # NOT gemma4 — the class name "Gemma4Client" is a legacy alias.
        assert "llama3.1" in DEFAULT_MODEL.lower()

    def test_02_base_url_is_local(self):
        # BASE_URL = http://127.0.0.1:11434 — accept either localhost or 127.0.0.1
        url = BASE_URL.lower()
        assert "127.0.0.1" in url or "localhost" in url

    def test_03_default_temp_low(self):
        assert DEFAULT_TEMP <= 0.2

    def test_04_max_retries_is_one(self):
        # 2026-06-05: MAX_RETRIES is 1 (fast-fail P0 fix), not 3.
        assert MAX_RETRIES == 1

    def test_05_seed_is_42(self):
        # C5: seed=42 everywhere.
        assert SEED == 42


class TestInstantiation:

    def test_06_creates_with_defaults(self, client):
        assert client.model    == DEFAULT_MODEL
        assert client.base_url == BASE_URL.rstrip("/")
        assert client.seed     == 42

    def test_07_custom_model(self):
        c = Gemma4Client(model="llama3:8b")
        assert c.model == "llama3:8b"

    def test_08_circuit_starts_closed(self, client):
        assert client._circuit_open  is False
        assert client._failure_count == 0

    def test_09_stats_start_at_zero(self, client):
        assert client._total_calls    == 0
        assert client._total_failures == 0


class TestChatMethod:

    def test_10_chat_returns_string(self, client):
        mock = _mock_ollama("Total net sales were $383.3 billion.")
        with patch("urllib.request.urlopen", return_value=mock):
            result = client.chat("What was Apple revenue?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_11_chat_returns_correct_content(self, client):
        mock = _mock_ollama("Answer: $383,285 million")
        with patch("urllib.request.urlopen", return_value=mock):
            result = client.chat("Revenue question")
        assert "383,285" in result

    def test_12_chat_increments_total_calls(self, client):
        mock = _mock_ollama("response")
        with patch("urllib.request.urlopen", return_value=mock):
            client.chat("test")
        assert client._total_calls == 1

    def test_13_chat_returns_empty_on_connection_error(self, client):
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            result = client.chat("test")
        assert result == ""

    def test_14_circuit_trips_after_max_failures(self, client):
        # 2026-06-05: MAX_RETRIES is 1 — each chat() call is one failure,
        # so a single failed call should trip the circuit.
        with patch("urllib.request.urlopen", side_effect=ConnectionError("refused")):
            client.chat("test1")  # failure_count = 1 -> circuit trips
        assert client._circuit_open is True

    def test_15_circuit_open_returns_empty_immediately(self, client):
        # Manually trip circuit with future reset time so it stays open
        client._circuit_open    = True
        client._circuit_open_at = time.time() + 9999
        result = client.chat("test")
        assert result == ""
        # total_calls is NOT incremented when circuit is open (returns before count)
        assert client._total_calls == 0

    def test_16_reset_circuit_clears_state(self, client):
        client._circuit_open   = True
        client._failure_count  = 3
        client.reset_circuit()
        assert client._circuit_open  is False
        assert client._failure_count == 0


class TestChatJson:

    def test_17_chat_json_returns_dict(self, client):
        mock = _mock_ollama('{"answer": "383285", "unit": "million"}')
        with patch("urllib.request.urlopen", return_value=mock):
            result = client.chat_json("Give JSON answer")
        assert isinstance(result, dict)

    def test_18_chat_json_parses_correctly(self, client):
        mock = _mock_ollama('{"value": 383285, "currency": "USD"}')
        with patch("urllib.request.urlopen", return_value=mock):
            result = client.chat_json("JSON question")
        assert result.get("value") == 383285

    def test_19_chat_json_strips_markdown_fences(self, client):
        mock = _mock_ollama('```json\n{"key": "val"}\n```')
        with patch("urllib.request.urlopen", return_value=mock):
            result = client.chat_json("JSON fenced")
        assert result.get("key") == "val"

    def test_20_chat_json_returns_empty_on_failure(self, client):
        with patch("urllib.request.urlopen", side_effect=ConnectionError):
            result = client.chat_json("fail")
        assert result == {}


class TestHealthCheck:

    def test_21_health_check_returns_dict(self, client):
        with patch.object(client, "is_available", return_value=False):
            h = client.health_check()
        assert isinstance(h, dict)
        assert "model"     in h
        assert "available" in h

    def test_22_health_includes_stats(self, client):
        with patch.object(client, "is_available", return_value=True):
            h = client.health_check()
        assert "total_calls"    in h
        assert "total_failures" in h
        assert "circuit_open"   in h

    def test_23_is_available_false_when_offline(self, client):
        with patch("urllib.request.urlopen", side_effect=Exception("offline")):
            assert client.is_available() is False


class TestSingleton:

    def test_24_get_llm_client_returns_instance(self):
        c = get_llm_client()
        assert isinstance(c, Gemma4Client)

    def test_25_get_llm_client_returns_same_instance(self):
        c1 = get_llm_client()
        c2 = get_llm_client()
        assert c1 is c2