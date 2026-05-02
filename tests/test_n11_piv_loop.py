"""
tests/test_n11_piv_loop.py
Tests for N11 PIV Loop — Planner, Implementor, Validator, Controller
PDR-BAAAI-001 · Rev 1.0

These tests use a MockLLMClient that never calls Ollama.
Tests verify structure, logic, retry behaviour, and state integration.
"""

import pytest
from src.analysis.piv_loop import (
    PIVLoopController,
    StrategicPlanner,
    ContextImplementor,
    CuriousValidator,
    OllamaClient,
    PlannerOutput,
    ImplementorOutput,
    ValidatorVerdict,
    PIVResult,
    MAX_RETRIES,
    HITL_CONF_THRESHOLD,
    CONF_DECAY,
    ITERATION_CAP,
    run_analyst_pod,
)
from src.state.ba_state import BAState


# ── Mock LLM Client ───────────────────────────────────────────────────────────

class MockLLMClient:
    """
    Mock Ollama client for testing.
    Returns configurable responses without calling any server.
    """
    def __init__(self, responses: list = None, available: bool = True):
        self.responses   = responses or []
        self._call_count = 0
        self._available  = available

    def chat(self, prompt: str, temperature: float = 0.1) -> str:
        if self._call_count < len(self.responses):
            resp = self.responses[self._call_count]
        else:
            resp = self.responses[-1] if self.responses else ""
        self._call_count += 1
        return resp

    def is_available(self) -> bool:
        return self._available


MOCK_PLANNER_RESPONSE = """
CURIOSITY_Q1: The question asks for Apple total net sales in fiscal year 2023.
CURIOSITY_Q2: Net sales / total revenue — top line income statement figure.
CURIOSITY_Q3: Income Statement section, page 94 of the 10-K filing.
CURIOSITY_Q4: FY2023 ends September not December. Unit confusion millions vs billions.
CURIOSITY_Q5: Prior year comparison in same table for trend analysis.
CURIOSITY_Q6: Apple fiscal year ends in September. Non-GAAP adjustments possible.

ANALYSIS_PLAN: Look in INCOME_STATEMENT section for total net sales row FY2023.
RETRIEVAL_HINTS: total net sales, revenue, income statement
VALIDATION_CRITERIA: Answer must cite exact dollar amount with units and fiscal year.
"""

MOCK_IMPLEMENTOR_PASS = """
ANSWER: Apple total net sales were $383,285 million in fiscal year 2023 [INCOME_STATEMENT/P94: $383,285M].
COMPUTATION: N/A
CONFIDENCE: 0.95 because the figure is directly cited from the income statement.
CITATIONS: [INCOME_STATEMENT / PAGE 94: $383,285 million]
"""

MOCK_IMPLEMENTOR_FAIL = """
ANSWER: Apple revenue was approximately 383 billion dollars.
COMPUTATION: N/A
CONFIDENCE: 0.60 because units may be imprecise.
CITATIONS: [general knowledge]
"""

MOCK_IMPLEMENTOR_MISS = """
RETRIEVAL_MISS: The exact net sales figure for FY2023 is not present in the retrieved chunks.
"""

MOCK_VALIDATOR_PASS = """
V1_SCOPE: Result: PASS Reason: All parts answered.
V2_UNITS: Result: PASS Reason: Units stated as millions.
V3_SIGN: Result: PASS Reason: Positive revenue, correct.
V4_CITATION: Result: PASS Reason: Citation present and valid.
V5_FISCAL_YEAR: Result: PASS Reason: FY2023 correctly stated.
V6_CONSISTENCY: Result: PASS Reason: Single figure, consistent.
V7_COMPLETENESS: Result: PASS Reason: Complete answer provided.
V8_GROUNDING: Result: PASS Reason: All claims from context.

FINAL VERDICT:
VALIDATOR_PASS: ALL 8 checks are PASS
REJECT_REASONS:
RETRY_INSTRUCTIONS:
"""

MOCK_VALIDATOR_FAIL = """
V1_SCOPE: Result: PASS Reason: Question addressed.
V2_UNITS: Result: FAIL Reason: Units stated as billions but context shows millions.
V3_SIGN: Result: PASS Reason: Positive value correct.
V4_CITATION: Result: FAIL Reason: Citation says 'general knowledge' not a valid section.
V5_FISCAL_YEAR: Result: PASS Reason: FY2023 correct.
V6_CONSISTENCY: Result: PASS Reason: N/A.
V7_COMPLETENESS: Result: PASS Reason: Complete.
V8_GROUNDING: Result: FAIL Reason: Figure not traceable to retrieved chunk.

FINAL VERDICT:
VALIDATOR_REJECT: ANY check is FAIL
REJECT_REASONS: 1. V2_UNITS: units wrong. 2. V4_CITATION: invalid. 3. V8_GROUNDING: hallucination.
RETRY_INSTRUCTIONS: Re-state units as millions. Cite INCOME_STATEMENT/PAGE 94 explicitly.
"""


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "chunk_id":    "chunk_1",
        "text":        "Apple total net sales were $383,285 million in fiscal 2023.",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "rank":        1,
        "retriever":   "rrf_reranker",
    },
    {
        "chunk_id":    "chunk_2",
        "text":        "Net income was $96,995 million in fiscal year 2023.",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "rank":        2,
        "retriever":   "rrf_reranker",
    },
]


@pytest.fixture
def mock_pass_controller():
    """Controller that always PASSes on first attempt."""
    llm = MockLLMClient(responses=[
        MOCK_PLANNER_RESPONSE,
        MOCK_IMPLEMENTOR_PASS,
        MOCK_VALIDATOR_PASS,
    ])
    return PIVLoopController(llm_client=llm, pod_role="analyst")


@pytest.fixture
def mock_fail_then_pass_controller():
    """Controller that fails once then passes."""
    llm = MockLLMClient(responses=[
        MOCK_PLANNER_RESPONSE,   # Planner
        MOCK_IMPLEMENTOR_FAIL,   # Implementor attempt 1
        MOCK_VALIDATOR_FAIL,     # Validator rejects
        MOCK_IMPLEMENTOR_PASS,   # Implementor attempt 2
        MOCK_VALIDATOR_PASS,     # Validator passes
    ])
    return PIVLoopController(llm_client=llm, pod_role="analyst")


@pytest.fixture
def mock_always_fail_controller():
    """Controller that always fails — triggers low_confidence."""
    llm = MockLLMClient(responses=[
        MOCK_PLANNER_RESPONSE,
        MOCK_IMPLEMENTOR_FAIL,
        MOCK_VALIDATOR_FAIL,
        MOCK_IMPLEMENTOR_FAIL,
        MOCK_VALIDATOR_FAIL,
        MOCK_IMPLEMENTOR_FAIL,
        MOCK_VALIDATOR_FAIL,
        MOCK_IMPLEMENTOR_FAIL,
        MOCK_VALIDATOR_FAIL,
    ])
    return PIVLoopController(llm_client=llm, pod_role="analyst")


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_max_retries_is_3(self):
        assert MAX_RETRIES == 3

    def test_02_hitl_threshold(self):
        assert HITL_CONF_THRESHOLD == 0.65

    def test_03_iteration_cap(self):
        assert ITERATION_CAP == 5

    def test_04_conf_decay_defined(self):
        assert 0 in CONF_DECAY
        assert 1 in CONF_DECAY
        assert 2 in CONF_DECAY
        assert 3 in CONF_DECAY

    def test_05_conf_decay_decreasing(self):
        assert CONF_DECAY[0] >= CONF_DECAY[1] >= CONF_DECAY[2] >= CONF_DECAY[3]

    def test_06_conf_decay_retry3_is_070(self):
        assert CONF_DECAY[3] == 0.70


# ── Group 2: Data classes ─────────────────────────────────────────────────────

class TestDataClasses:

    def test_07_planner_output_fields(self):
        p = PlannerOutput(
            analysis_plan="plan", retrieval_hints=["h1"],
            validation_criteria="v", curiosity_answers={"Q1": "a"},
        )
        assert p.analysis_plan       == "plan"
        assert p.retrieval_hints     == ["h1"]
        assert p.validation_criteria == "v"

    def test_08_implementor_output_fields(self):
        i = ImplementorOutput(
            answer="ans", computation="N/A",
            confidence=0.9, citations=[], output_type="ANSWER",
        )
        assert i.output_type == "ANSWER"
        assert i.confidence  == 0.9

    def test_09_validator_verdict_fields(self):
        v = ValidatorVerdict(
            result="VALIDATOR_PASS",
            checks={"V1_SCOPE": "PASS"},
            reject_reasons=[],
            retry_instructions="",
        )
        assert v.result == "VALIDATOR_PASS"

    def test_10_piv_result_fields(self):
        r = PIVResult(
            answer="ans", confidence=0.9, citations=[],
            computation="N/A", retries_used=0, low_confidence=False,
            validator_checks={}, reject_reasons=[], pod_role="analyst",
        )
        assert r.pod_role == "analyst"
        assert r.low_confidence is False


# ── Group 3: OllamaClient ─────────────────────────────────────────────────────

class TestOllamaClient:

    def test_11_instantiates(self):
        client = OllamaClient()
        assert client is not None

    def test_12_default_model(self):
        client = OllamaClient()
        assert "llama" in client.model.lower() or "financebench" in client.model.lower()

    def test_13_unavailable_returns_false(self):
        """Non-existent server returns False without raising."""
        client = OllamaClient(base_url="http://localhost:19999", timeout=1)
        assert client.is_available() is False

    def test_14_mock_client_works(self):
        mock = MockLLMClient(responses=["Hello"])
        assert mock.chat("test") == "Hello"
        assert mock.is_available() is True


# ── Group 4: StrategicPlanner ─────────────────────────────────────────────────

class TestStrategicPlanner:

    def test_15_planner_returns_output(self):
        llm     = MockLLMClient(responses=[MOCK_PLANNER_RESPONSE])
        planner = StrategicPlanner(llm)
        result  = planner.run("What was net income?", SAMPLE_CHUNKS)
        assert isinstance(result, PlannerOutput)

    def test_16_planner_has_analysis_plan(self):
        llm     = MockLLMClient(responses=[MOCK_PLANNER_RESPONSE])
        planner = StrategicPlanner(llm)
        result  = planner.run("What was net income?", SAMPLE_CHUNKS)
        assert len(result.analysis_plan) > 0

    def test_17_planner_has_curiosity_answers(self):
        llm     = MockLLMClient(responses=[MOCK_PLANNER_RESPONSE])
        planner = StrategicPlanner(llm)
        result  = planner.run("What was net income?", SAMPLE_CHUNKS)
        assert isinstance(result.curiosity_answers, dict)

    def test_18_planner_fallback_when_llm_unavailable(self):
        llm     = MockLLMClient(responses=[""])
        planner = StrategicPlanner(llm)
        result  = planner.run("What was net income?", SAMPLE_CHUNKS)
        assert isinstance(result, PlannerOutput)
        assert len(result.analysis_plan) > 0


# ── Group 5: ContextImplementor ──────────────────────────────────────────────

class TestContextImplementor:

    def test_19_implementor_returns_output(self):
        llm   = MockLLMClient(responses=[MOCK_IMPLEMENTOR_PASS])
        impl  = ContextImplementor(llm)
        result = impl.run("net income?", "context text", "plan", "criteria")
        assert isinstance(result, ImplementorOutput)

    def test_20_implementor_parses_answer(self):
        llm    = MockLLMClient(responses=[MOCK_IMPLEMENTOR_PASS])
        impl   = ContextImplementor(llm)
        result = impl.run("net income?", "context", "plan", "criteria")
        assert result.output_type == "ANSWER"
        assert len(result.answer) > 0

    def test_21_implementor_detects_retrieval_miss(self):
        llm    = MockLLMClient(responses=[MOCK_IMPLEMENTOR_MISS])
        impl   = ContextImplementor(llm)
        result = impl.run("net income?", "context", "plan", "criteria")
        assert result.output_type == "RETRIEVAL_MISS"

    def test_22_confidence_in_range(self):
        llm    = MockLLMClient(responses=[MOCK_IMPLEMENTOR_PASS])
        impl   = ContextImplementor(llm)
        result = impl.run("net income?", "context", "plan", "criteria", retry_count=0)
        assert 0.0 <= result.confidence <= 1.0

    def test_23_confidence_decays_on_retry(self):
        llm   = MockLLMClient(responses=[MOCK_IMPLEMENTOR_PASS, MOCK_IMPLEMENTOR_PASS])
        impl  = ContextImplementor(llm)
        r0    = impl.run("q", "ctx", "plan", "crit", retry_count=0)
        r2    = impl.run("q", "ctx", "plan", "crit", retry_count=2)
        assert r2.confidence <= r0.confidence

    def test_24_fallback_when_llm_unavailable(self):
        llm    = MockLLMClient(responses=[""])
        impl   = ContextImplementor(llm)
        result = impl.run("q", "ctx", "plan", "crit")
        assert isinstance(result, ImplementorOutput)


# ── Group 6: CuriousValidator ─────────────────────────────────────────────────

class TestCuriousValidator:

    def test_25_validator_returns_verdict(self):
        llm       = MockLLMClient(responses=[MOCK_VALIDATOR_PASS])
        validator = CuriousValidator(llm)
        result    = validator.run("q", "answer", "context", "criteria")
        assert isinstance(result, ValidatorVerdict)

    def test_26_validator_pass(self):
        llm       = MockLLMClient(responses=[MOCK_VALIDATOR_PASS])
        validator = CuriousValidator(llm)
        result    = validator.run("q", "answer", "context", "criteria")
        assert result.result == "VALIDATOR_PASS"

    def test_27_validator_fail(self):
        llm       = MockLLMClient(responses=[MOCK_VALIDATOR_FAIL])
        validator = CuriousValidator(llm)
        result    = validator.run("q", "answer", "context", "criteria")
        assert result.result == "VALIDATOR_REJECT"

    def test_28_validator_fail_has_reasons(self):
        llm       = MockLLMClient(responses=[MOCK_VALIDATOR_FAIL])
        validator = CuriousValidator(llm)
        result    = validator.run("q", "answer", "context", "criteria")
        if result.result == "VALIDATOR_REJECT":
            assert len(result.reject_reasons) > 0 or len(result.retry_instructions) > 0

    def test_29_validator_has_8_checks(self):
        llm       = MockLLMClient(responses=[MOCK_VALIDATOR_PASS])
        validator = CuriousValidator(llm)
        result    = validator.run("q", "answer", "context", "criteria")
        assert len(result.checks) == 8

    def test_30_validator_fallback_when_llm_unavailable(self):
        llm       = MockLLMClient(responses=[""])
        validator = CuriousValidator(llm)
        result    = validator.run("q", "answer", "context", "criteria")
        assert result.result == "VALIDATOR_PASS"


# ── Group 7: PIVLoopController ────────────────────────────────────────────────

class TestPIVLoopController:

    def test_31_pass_on_first_attempt(self, mock_pass_controller):
        result = mock_pass_controller.run_piv(
            query  = "What was net income FY2023?",
            chunks = SAMPLE_CHUNKS,
        )
        assert isinstance(result, PIVResult)
        assert result.retries_used == 0
        assert result.low_confidence is False

    def test_32_retry_then_pass(self, mock_fail_then_pass_controller):
        result = mock_fail_then_pass_controller.run_piv(
            query  = "What was net income?",
            chunks = SAMPLE_CHUNKS,
        )
        assert isinstance(result, PIVResult)
        assert result.retries_used >= 1

    def test_33_exhausted_returns_low_confidence(self, mock_always_fail_controller):
        result = mock_always_fail_controller.run_piv(
            query  = "What was net income?",
            chunks = SAMPLE_CHUNKS,
        )
        assert result.low_confidence is True

    def test_34_result_has_answer(self, mock_pass_controller):
        result = mock_pass_controller.run_piv(
            query  = "net income",
            chunks = SAMPLE_CHUNKS,
        )
        assert isinstance(result.answer, str)

    def test_35_result_has_confidence(self, mock_pass_controller):
        result = mock_pass_controller.run_piv(
            query  = "net income",
            chunks = SAMPLE_CHUNKS,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_36_result_has_pod_role(self, mock_pass_controller):
        result = mock_pass_controller.run_piv(
            query  = "net income",
            chunks = SAMPLE_CHUNKS,
        )
        assert result.pod_role == "analyst"

    def test_37_retries_never_exceed_max(self, mock_always_fail_controller):
        result = mock_always_fail_controller.run_piv(
            query  = "net income",
            chunks = SAMPLE_CHUNKS,
        )
        assert result.retries_used <= MAX_RETRIES + 1

    def test_38_empty_query_still_returns_result(self):
        llm    = MockLLMClient(responses=[MOCK_PLANNER_RESPONSE, MOCK_IMPLEMENTOR_PASS, MOCK_VALIDATOR_PASS])
        ctrl   = PIVLoopController(llm_client=llm)
        result = ctrl.run_piv(query="", chunks=[])
        assert isinstance(result, PIVResult)


# ── Group 8: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_39_run_writes_analyst_output(self, mock_pass_controller):
        state = BAState(
            session_id        = "t39",
            query             = "What was net income FY2023?",
            query_type        = "numerical",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pass_controller.run(state)
        assert hasattr(state, "analyst_output")
        assert isinstance(state.analyst_output, str)

    def test_40_run_writes_confidence(self, mock_pass_controller):
        state = BAState(
            session_id        = "t40",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pass_controller.run(state)
        assert 0.0 <= state.analyst_confidence <= 1.0

    def test_41_run_writes_piv_round(self, mock_pass_controller):
        state = BAState(
            session_id        = "t41",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pass_controller.run(state)
        assert hasattr(state, "piv_round")
        assert isinstance(state.piv_round, int)

    def test_42_seed_unchanged_after_run(self, mock_pass_controller):
        """C5: seed must remain 42."""
        state = BAState(
            session_id        = "t42",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pass_controller.run(state)
        assert state.seed == 42

    def test_43_low_confidence_triggers_hitl_flag(
        self, mock_always_fail_controller
    ):
        state = BAState(
            session_id        = "t43",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_always_fail_controller.run(state)
        assert state.low_confidence is True

    def test_44_empty_query_skips_piv(self):
        llm   = MockLLMClient(responses=[])
        ctrl  = PIVLoopController(llm_client=llm)
        state = BAState(session_id="t44", query="")
        state = ctrl.run(state)
        assert state.analyst_output  == ""
        assert state.analyst_confidence == 0.0

    def test_45_no_rlef_in_analyst_output(self, mock_pass_controller):
        """C9: analyst_output must never contain _rlef_ fields."""
        state = BAState(
            session_id        = "t45",
            query             = "net income",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        state = mock_pass_controller.run(state)
        assert "_rlef_" not in state.analyst_output


# ── Group 9: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_46_run_analyst_pod_returns_state(self):
        llm   = MockLLMClient(responses=[
            MOCK_PLANNER_RESPONSE,
            MOCK_IMPLEMENTOR_PASS,
            MOCK_VALIDATOR_PASS,
        ])
        state = BAState(
            session_id        = "t46",
            query             = "What was net income?",
            retrieval_stage_2 = SAMPLE_CHUNKS,
        )
        result = run_analyst_pod(state, llm_client=llm)
        assert hasattr(result, "analyst_output")
        assert result.seed == 42

    # ════════════════════════════════════════════════════════════════════════════
# BUG #4 — PIV early-exit on empty retrieval
# ════════════════════════════════════════════════════════════════════════════

class TestBug4EarlyExit:
    """Regression tests for Bug #4: pointless retries on empty retrieval.

    Before fix: empty chunks → Planner runs → Implementor RETRIEVAL_MISS
                → loop continues for full retry budget → ~3-9 min wasted.
    After fix:  empty chunks → return immediately (~0.1s).
    """

    def _build_controller(self):
        """Build PIVLoopController and patch its sub-agents with mocks."""
        from unittest.mock import MagicMock
        ctrl = PIVLoopController(
            llm_client = MockLLMClient(responses=[""]),
            pod_role   = "analyst",
        )
        # Patch sub-agents AFTER construction so we can count calls
        ctrl.planner     = MagicMock()
        ctrl.implementor = MagicMock()
        ctrl.validator   = MagicMock()
        return ctrl

    def test_empty_chunks_exits_without_calling_planner(self):
        """Bug #4: chunks=[] must exit BEFORE Planner runs."""
        ctrl = self._build_controller()

        result = ctrl.run_piv(
            query            = "What was net income?",
            chunks           = [],
            query_type       = "text",
            query_difficulty = "medium",
        )

        # Critical: Planner, Implementor, Validator must NEVER be called
        assert ctrl.planner.run.call_count     == 0, (
            f"Bug #4 regression: Planner called "
            f"{ctrl.planner.run.call_count}× on empty chunks (must be 0)"
        )
        assert ctrl.implementor.run.call_count == 0
        assert ctrl.validator.run.call_count   == 0

        assert result.retries_used   == 0
        assert result.low_confidence is True
        assert result.confidence     == 0.0
        assert "RETRIEVAL_MISS" in result.answer

    def test_empty_chunks_returns_quickly(self):
        """Bug #4: empty chunks path must return without blocking."""
        import time
        ctrl = self._build_controller()
        t0 = time.time()
        result = ctrl.run_piv(
            query="anything",
            chunks=[],
            query_type="text",
            query_difficulty="medium",
        )
        elapsed = time.time() - t0
        assert elapsed < 1.0, (
            f"Bug #4: empty-chunks path took {elapsed:.2f}s "
            f"(must be <1s; previously was 3-9 min)"
        )

    def test_empty_chunks_returns_low_confidence(self):
        """Bug #4: early exit must mark low_confidence=True."""
        ctrl = self._build_controller()
        result = ctrl.run_piv(
            query="anything",
            chunks=[],
            query_type="text",
            query_difficulty="medium",
        )
        assert result.retries_used == 0
        assert result.low_confidence is True
        assert result.confidence == 0.0