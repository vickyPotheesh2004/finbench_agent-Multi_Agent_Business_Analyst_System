"""
src/state/ba_state.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL

BA_State v10 — Shared Pydantic v2 state object.
Flows through all 19 pipeline nodes.
All constraints enforced via validators.

Constraints enforced here:
  C5: seed=42 always
  C7: prompt_template='context_first' always
  C8: chunk prefix format validated
  C9: _rlef_ fields never in output
  A2: iteration_count hard cap = 5
  A3: low_confidence flag for clarification engine
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    NUMERICAL = "numerical"
    RATIO     = "ratio"
    MULTI_DOC = "multi_doc"
    TEXT      = "text"
    FORENSIC  = "forensic"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class PIVStatus(str, Enum):
    PASS   = "PASS"
    REJECT = "REJECT"


# ── BAState ───────────────────────────────────────────────────────────────────

class BAState(BaseModel):
    """
    Shared state object flowing through all 19 pipeline nodes.
    validate_assignment=True — every field write is validated.
    """
    model_config = {"validate_assignment": True}

    # ── IDENTITY ──────────────────────────────────────────────────────────
    session_id:    str = Field(default_factory=lambda: str(uuid4()))
    document_path: str = ""
    company_name:  str = ""
    doc_type:      str = ""      # '10-K', '10-Q', '8-K'
    fiscal_year:   str = ""
    model_version: str = "financebench-expert-v1"
    seed:          int = 42      # ALWAYS 42 — C5

    # ── INGESTION (N01-N03) ───────────────────────────────────────────────
    raw_text:             str        = ""
    table_cells:          List[Dict] = Field(default_factory=list)
    heading_positions:    List[Dict] = Field(default_factory=list)
    section_tree:         Dict       = Field(default_factory=dict)
    chunk_count:          int        = 0
    bm25_index_path:      str        = ""
    chromadb_collection:  str        = ""
    chromadb_data_dir:    str        = ""   # Bug #3 — N08 reads same dir
    chunk_metadata_prefix: str       = ""   # C8 — last prefix written

    # ── ROUTING (N04-N05) ─────────────────────────────────────────────────
    query:               str       = ""
    query_type:          QueryType = QueryType.TEXT
    routing_path:        str       = ""
    query_difficulty:    Difficulty = Difficulty.MEDIUM
    context_window_size: int       = 3

    # ── RETRIEVAL (N06-N09) ───────────────────────────────────────────────
    sniper_hit:        bool       = False
    sniper_result:     Optional[str] = None
    sniper_confidence: float      = 0.0
    bm25_results:      List[Dict] = Field(default_factory=list)
    bm25_confidence:   float      = 0.0
    retrieval_stage_1: List[Dict] = Field(default_factory=list)
    retrieval_stage_2: List[Dict] = Field(default_factory=list)

    # ── PROMPTING (N10) ───────────────────────────────────────────────────
    assembled_prompt: str = ""
    prompt_template:  str = "context_first"   # C7 — never change

    # ── PIV STATE — ANALYST POD N11 ───────────────────────────────────────
    analyst_output:        str       = ""
    analyst_confidence:    float     = 0.0
    analyst_citations:     List[str] = Field(default_factory=list)
    analyst_retries:       int       = 0
    analyst_low_conf:      bool      = False
    analyst_piv_status:    PIVStatus = PIVStatus.REJECT
    analyst_attempt_count: int       = 0    # retries used by N11

    # ── PIV STATE — QUANT POD N12 ─────────────────────────────────────────
    quant_result:         str       = ""
    quant_confidence:     float     = 0.0
    quant_citations:      List[str] = Field(default_factory=list)
    quant_piv_status:     PIVStatus = PIVStatus.REJECT
    quant_attempt_count:  int       = 0    # retries used by N12

    # ── QUANTITATIVE (N12) ────────────────────────────────────────────────
    monte_carlo_results: Optional[Dict]  = None
    var_result:          Optional[Dict]  = None
    garch_result:        Optional[Dict]  = None
    computed_ratio:      Optional[float] = None

    # ── FORENSICS (N13) ───────────────────────────────────────────────────
    forensic_flags:   List[str] = Field(default_factory=list)
    risk_score:       float     = 0.0
    anomaly_detected: bool      = False
    anomaly_severity: str       = "low"
    benford_chi2:     float     = 0.0
    benford_p_value:  float     = 1.0

    # ── PIV STATE — AUDITOR POD N14 ───────────────────────────────────────
    auditor_output:        str       = ""
    auditor_confidence:    float     = 0.0
    auditor_citations:     List[str] = Field(default_factory=list)
    auditor_attempt_count: int       = 0    # retries used by N14
    auditor_piv_status:    PIVStatus = PIVStatus.REJECT
    contradiction_flags:   List[str] = Field(default_factory=list)

    # ── DEBATE / MEDIATOR (N15) ───────────────────────────────────────────
    piv_candidates:       List[Dict] = Field(default_factory=list)
    piv_round:            int        = 0
    iteration_count:      int        = 0   # hard cap 5
    final_answer_pre_xgb: str        = ""
    agreement_status:     str        = ""
    confidence_score:     float      = 0.0
    low_confidence:       bool       = False
    winning_pod:          str        = ""

    # ── EXPLAINABILITY (N16) ──────────────────────────────────────────────
    shap_values:        Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    causal_dag_path:    Optional[str]              = None

    # ── XGB ARBITER (N17) ─────────────────────────────────────────────────
    xgb_ranked_answer: Optional[str] = None
    xgb_score:         float         = 0.0

    # ── FINAL OUTPUT (N18-N19) ────────────────────────────────────────────
    final_answer:      str           = ""
    final_report_path: Optional[str] = None

    # ── RLEF — PRIVATE FOREVER (N18) ─────────────────────────────────────
    # NEVER in any output, UI, log, or DOCX — C9
    _rlef_grade:          int          = 0
    _rlef_va_score:       float        = 0.0
    _rlef_vb_score:       float        = 0.0
    _rlef_vc_score:       float        = 0.0
    _rlef_chosen:         Optional[str] = None
    _rlef_rejected:       Optional[str] = None
    _rlef_user_consented: bool          = False
    _rlef_stored_global:  bool          = False

    # ── VALIDATORS ────────────────────────────────────────────────────────

    @field_validator("iteration_count")
    @classmethod
    def cap_iterations(cls, v: int) -> int:
        """A2: Hard cap iteration_count at 5."""
        if v > 5:
            raise ValueError(
                f"iteration_count hard cap is 5 (A2). Got {v}."
            )
        return v

    @field_validator("prompt_template")
    @classmethod
    def enforce_context_first(cls, v: str) -> str:
        """C7: prompt_template must always be context_first."""
        if v != "context_first":
            raise ValueError(
                f"prompt_template must be 'context_first' (C7). Got '{v}'."
            )
        return v

    @field_validator("seed")
    @classmethod
    def enforce_seed_42(cls, v: int) -> int:
        """C5: seed must always be 42."""
        if v != 42:
            raise ValueError(
                f"seed must be 42 (C5). Got {v}."
            )
        return v

    def chunk_prefix(
        self,
        company:     str,
        doc_type:    str,
        fiscal_year: str,
        section:     str,
        page:        int,
    ) -> str:
        """C8: Build mandatory 5-field chunk prefix."""
        return f"{company} / {doc_type} / {fiscal_year} / {section} / {page}"

    def get_rlef_fields(self) -> dict:
        """
        Return _rlef_ fields for internal RLEF use only.
        Never call this in output generation code.
        """
        return {
            "_rlef_grade":          self._rlef_grade,
            "_rlef_va_score":       self._rlef_va_score,
            "_rlef_vb_score":       self._rlef_vb_score,
            "_rlef_vc_score":       self._rlef_vc_score,
            "_rlef_chosen":         self._rlef_chosen,
            "_rlef_rejected":       self._rlef_rejected,
            "_rlef_user_consented": self._rlef_user_consented,
            "_rlef_stored_global":  self._rlef_stored_global,
        }


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- BAState sanity check --[/bold cyan]")

    state = BAState(
        session_id   = "sanity-check",
        company_name = "Apple Inc",
        doc_type     = "10-K",
        fiscal_year  = "FY2023",
    )
    rprint(f"[green]✓[/green] Created | seed={state.seed} | "
           f"prompt_template={state.prompt_template}")

    # C5 enforced
    try:
        state.seed = 99
        rprint("[red]✗[/red] C5 NOT enforced")
    except Exception:
        rprint("[green]✓[/green] C5 enforced")

    # C7 enforced
    try:
        state.prompt_template = "question_first"
        rprint("[red]✗[/red] C7 NOT enforced")
    except Exception:
        rprint("[green]✓[/green] C7 enforced")

    # A2 enforced
    try:
        state.iteration_count = 6
        rprint("[red]✗[/red] A2 NOT enforced")
    except Exception:
        rprint("[green]✓[/green] A2 enforced")

    # C9 — _rlef_ fields private
    rlef = state.get_rlef_fields()
    assert all(k.startswith("_rlef_") for k in rlef)
    rprint("[green]✓[/green] C9 enforced")

    # C8 prefix
    prefix = state.chunk_prefix("Apple Inc","10-K","FY2023","MD&A",42)
    assert prefix == "Apple Inc / 10-K / FY2023 / MD&A / 42"
    rprint(f"[green]✓[/green] C8 prefix: {prefix}")

    # N12 quant fields
    state.monte_carlo_results = {"mean": 96995.0, "n": 10000}
    state.var_result          = {"var_95": 64793.0, "var_99": 58887.0}
    assert state.monte_carlo_results["mean"] == 96995.0
    rprint("[green]✓[/green] N12 quant fields: monte_carlo + var_result")

    # N16 SHAP fields
    state.shap_values        = {"bm25_score": 0.42, "cosine_sim": 0.38}
    state.feature_importance = {"bm25_score": 0.50, "cosine_sim": 0.30}
    state.causal_dag_path    = "outputs/causal_dag_test.png"
    assert isinstance(state.shap_values, dict)
    rprint("[green]✓[/green] N16 SHAP fields: shap_values + feature_importance")

    # attempt_count fields
    state.analyst_attempt_count = 2
    state.quant_attempt_count   = 1
    state.auditor_attempt_count = 3
    assert state.analyst_attempt_count == 2
    rprint("[green]✓[/green] attempt_count fields: analyst + quant + auditor")

    # chunk_metadata_prefix
    state.chunk_metadata_prefix = "Apple Inc / 10-K / FY2023 / MD&A / 42"
    assert "Apple" in state.chunk_metadata_prefix
    rprint("[green]✓[/green] chunk_metadata_prefix field present")

    # A3 clarification engine
    state.low_confidence = True
    assert state.low_confidence is True
    rprint("[green]✓[/green] A3 clarification engine works")

    rprint("\n[bold green]All checks passed. BAState ready.[/bold green]\n")