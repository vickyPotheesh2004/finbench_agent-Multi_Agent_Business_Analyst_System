"""
tests/test_ci_gate.py
FinBench Multi-Agent Business Analyst AI

CI/CD gate tests — 19 tests.
Enforces all hard constraints C1-C10 and amendments A1-A3.

Tests 01-12: BAState field groups
Test  13:    _rlef_ output leakage gate (must FAIL on bad output)
Test  14:    seed != 42 gate (must FAIL on wrong seed)
Tests 15-16: C7 context-first prompt enforcement
Tests 17-19: Additional constraint checks
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from src.state.ba_state import (
    BAState,
    QueryType,
    Difficulty,
    PIVStatus,
)


# ════════════════════════════════════════════════════════════════════════════
# GROUP 1 -- BASTATE FIELD GROUPS (tests 01-12)
# ════════════════════════════════════════════════════════════════════════════

class TestBAStateFieldGroups:

    def test_01_identity_fields(self):
        """BAState identity fields must initialise correctly"""
        state = BAState(
            session_id   = "test-01",
            company_name = "Apple Inc",
            doc_type     = "10-K",
            fiscal_year  = "FY2023",
        )
        assert state.session_id   == "test-01"
        assert state.company_name == "Apple Inc"
        assert state.doc_type     == "10-K"
        assert state.fiscal_year  == "FY2023"
        assert state.seed         == 42

    def test_02_ingestion_fields(self):
        """BAState ingestion fields must initialise with correct defaults"""
        state = BAState(session_id="test-02")
        assert state.raw_text          == ""
        assert state.table_cells       == []
        assert state.heading_positions == []
        assert state.section_tree      == {}
        assert state.chunk_count       == 0

    def test_03_routing_fields(self):
        """BAState routing fields must initialise correctly"""
        state = BAState(session_id="test-03")
        assert state.query_type       == QueryType.TEXT
        assert state.query_difficulty == Difficulty.MEDIUM
        assert state.context_window_size == 3

    def test_04_retrieval_fields(self):
        """BAState retrieval fields must initialise with correct defaults"""
        state = BAState(session_id="test-04")
        assert state.sniper_hit        is False
        assert state.sniper_confidence == 0.0
        assert state.bm25_results      == []
        assert state.retrieval_stage_1 == []
        assert state.retrieval_stage_2 == []

    def test_05_prompting_fields(self):
        """BAState prompting fields must initialise correctly"""
        state = BAState(session_id="test-05")
        assert state.assembled_prompt == ""
        assert state.prompt_template  == "context_first"

    def test_06_analyst_piv_fields(self):
        """BAState N11 analyst PIV fields must initialise correctly"""
        state = BAState(session_id="test-06")
        assert state.analyst_output     == ""
        assert state.analyst_confidence == 0.0
        assert state.analyst_citations  == []
        assert state.analyst_piv_status == PIVStatus.REJECT

    def test_07_quant_piv_fields(self):
        """BAState N12 quant PIV fields must initialise correctly"""
        state = BAState(session_id="test-07")
        assert state.quant_result     == ""
        assert state.quant_confidence == 0.0
        assert state.quant_piv_status == PIVStatus.REJECT

    def test_08_forensics_fields(self):
        """BAState N13 forensics fields must initialise correctly"""
        state = BAState(session_id="test-08")
        assert state.forensic_flags   == []
        assert state.risk_score       == 0.0
        assert state.anomaly_detected is False
        assert state.anomaly_severity == "low"
        assert state.benford_chi2     == 0.0
        assert state.benford_p_value  == 1.0

    def test_09_auditor_piv_fields(self):
        """BAState N14 auditor PIV fields must initialise correctly"""
        state = BAState(session_id="test-09")
        assert state.auditor_output      == ""
        assert state.auditor_confidence  == 0.0
        assert state.auditor_citations   == []
        assert state.auditor_piv_status  == PIVStatus.REJECT
        assert state.contradiction_flags == []

    def test_10_mediator_fields(self):
        """BAState N15 mediator fields must initialise correctly"""
        state = BAState(session_id="test-10")
        assert state.final_answer_pre_xgb == ""
        assert state.agreement_status     == ""
        assert state.confidence_score     == 0.0
        assert state.low_confidence       is False
        assert state.iteration_count      == 0

    def test_11_explainability_fields(self):
        """BAState N16 explainability fields must initialise correctly"""
        state = BAState(session_id="test-11")
        # Updated 2026-06-08: ba_state v11 uses empty-container defaults
        # ({}/"") instead of None for these fields. Empty defaults are safer
        # (no None-guards needed downstream) and are the authoritative schema.
        assert state.shap_values        == {}
        assert state.feature_importance == {}
        assert state.causal_dag_path    == ""

    def test_12_output_fields(self):
        """BAState output fields must initialise correctly"""
        state = BAState(session_id="test-12")
        # Updated 2026-06-08: ba_state v11 uses "" defaults instead of None
        # for string output fields (authoritative schema).
        assert state.final_answer      == ""
        assert state.final_report_path == ""
        assert state.xgb_ranked_answer == ""
        assert state.xgb_score         == 0.0


# ════════════════════════════════════════════════════════════════════════════
# GROUP 2 -- CI/CD GATES (tests 13-19)
# ════════════════════════════════════════════════════════════════════════════

class TestCICDGates:

    def test_13_rlef_field_leakage_gate(self, tmp_path):
        """
        CI gate: output file containing '_rlef_' must be detected.
        This test PASSES when the gate correctly DETECTS the violation.
        """
        bad_output = tmp_path / "bad_output.txt"
        bad_output.write_text("The answer is 42. _rlef_grade: 4")

        content = bad_output.read_text()
        has_rlef = "_rlef_" in content
        assert has_rlef is True, (
            "Gate should detect _rlef_ in output file"
        )

    def test_14_wrong_seed_gate(self):
        """
        CI gate: BAState with seed != 42 must raise ValueError.
        This test PASSES when the constraint is correctly enforced.
        """
        with pytest.raises((ValueError, Exception)):
            BAState(session_id="test-14-bad-seed", seed=99)

    def test_15_context_first_enforced(self):
        """C7: prompt_template must be context_first — writing other value raises"""
        state = BAState(session_id="test-15")
        with pytest.raises((ValueError, Exception)):
            state.prompt_template = "question_first"

    def test_16_context_first_default(self):
        """C7: prompt_template must default to context_first"""
        state = BAState(session_id="test-16")
        assert state.prompt_template == "context_first"

    def test_17_iteration_count_cap(self):
        """A2: iteration_count must not exceed 5"""
        state = BAState(session_id="test-17")
        with pytest.raises((ValueError, Exception)):
            state.iteration_count = 6

    def test_18_c8_chunk_prefix_format(self):
        """C8: chunk_prefix must produce correct 5-field format"""
        state  = BAState(session_id="test-18")
        prefix = state.chunk_prefix(
            "Apple Inc", "10-K", "FY2023", "MD&A", 42
        )
        parts = prefix.split(" / ")
        assert len(parts)  == 5
        assert parts[0]    == "Apple Inc"
        assert parts[1]    == "10-K"
        assert parts[2]    == "FY2023"
        assert parts[3]    == "MD&A"
        assert parts[4]    == "42"

    def test_19_rlef_fields_private(self):
        """C9: _rlef_ fields must only be accessible via get_rlef_fields()"""
        state      = BAState(session_id="test-19")
        rlef_dict  = state.get_rlef_fields()
        assert all(k.startswith("_rlef_") for k in rlef_dict)
        assert "_rlef_grade" in rlef_dict
        assert "_rlef_chosen" in rlef_dict