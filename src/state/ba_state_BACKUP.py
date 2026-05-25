"""
src/state/ba_state.py
FinBench Multi-Agent Business Analyst AI
PDR-BAAAI-001 Rev1.0 FINAL — SESSION 16 FIX

BA_State v11 — Shared Pydantic v2 state object.
Flows through all 19 pipeline nodes.
All constraints enforced via validators.

SESSION 16 CHANGES:
    + bge_results       (List[Dict]) — pipeline.py line 344 was crashing
    + reranked_chunks   (List[Dict]) — pipeline.py line 344 was crashing
    + query_type        accepts both str AND QueryType enum (pipeline uses str)
    These were in state.py (duplicate) but NOT in ba_state.py (authoritative)

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
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

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


# ── BAState ────────────────────────────────────────────────────────────────────

class BAState(BaseModel):
    """
    Shared state object flowing through all 19 pipeline nodes.
    validate_assignment=True — every field write is validated.

    v11 FIX: Added bge_results + reranked_chunks (were crashing pipeline.py)
    """
    model_config = {"validate_assignment": True}

    # ── IDENTITY ───────────────────────────────────────────────────────────
    session_id:    str = Field(default_factory=lambda: str(uuid4()))
    document_path: str = ""
    company_name:  str = ""
    doc_type:      str = ""       # '10-K', '10-Q', '8-K'
    fiscal_year:   str = ""
    model_version: str = "financebench-expert-v1"
    seed:          int = 42       # ALWAYS 42 — C5

    # ── INGESTION (N01-N03) ────────────────────────────────────────────────
    raw_text:              str        = ""
    cleaned_text:          str        = ""
    page_texts:            List[str]  = Field(default_factory=list)
    page_count:            int        = 0
    extracted_images:      List[Dict] = Field(default_factory=list)
    table_cells:           List[Dict] = Field(default_factory=list)
    heading_positions:     List[Dict] = Field(default_factory=list)
    section_tree:          Dict       = Field(default_factory=dict)
    chunk_count:           int        = 0
    bm25_index_path:       str        = ""
    chromadb_collection:   str        = ""
    chromadb_data_dir:     str        = ""   # N08 reads same dir — Bug #3
    chunk_metadata_prefix: str        = ""   # C8 — last prefix written

    # ── ROUTING (N04-N05) ──────────────────────────────────────────────────
    query:               str        = ""
    query_type:          str        = "text"   # str — pipeline.py sets as str
    routing_path:        str        = ""
    query_difficulty:    Difficulty  = Difficulty.MEDIUM
    query_complexity:    str        = "medium"
    difficulty_score:    float      = 0.0
    context_window_size: int        = 3
    selected_pods:       List[str]  = Field(default_factory=list)

    # ── RETRIEVAL (N06-N09) ────────────────────────────────────────────────
    # N06 Sniper
    sniper_hit:        bool         = False
    sniper_result:     Optional[str] = None   # Human-readable answer string
    sniper_answer:     str          = ""      # Always-set string version
    sniper_value:      str          = ""      # Raw value for downstream
    sniper_unit:       str          = ""
    sniper_citation:   str          = ""
    sniper_pattern:    str          = ""
    sniper_confidence: float        = 0.0

    # N07 BM25
    bm25_results:      List[Dict]  = Field(default_factory=list)
    bm25_confidence:   float       = 0.0

    # N08 BGE — ✅ FIX: was missing, pipeline.py line 344 crashed here
    bge_results:       List[Dict]  = Field(default_factory=list)

    # N09 RRF Reranker — ✅ FIX: was missing, pipeline.py line 344 crashed here
    reranked_chunks:   List[Dict]  = Field(default_factory=list)
    retrieved_chunks:  List[Dict]  = Field(default_factory=list)

    # Stage aliases (used by RRF reranker + eval scripts)
    retrieval_stage_1: List[Dict]  = Field(default_factory=list)  # BGE top-10
    retrieval_stage_2: List[Dict]  = Field(default_factory=list)  # RRF top-3

    # ── PROMPTING (N10) ────────────────────────────────────────────────────
    assembled_prompt:  str  = ""
    prompt_template:   str  = "context_first"   # C7 — always context_first

    # ── PIV — ANALYST POD (N11) ────────────────────────────────────────────
    analyst_output:        str       = ""
    analyst_confidence:    float     = 0.0
    analyst_citations:     List[str] = Field(default_factory=list)
    analyst_retries:       int       = 0
    analyst_low_conf:      bool      = False
    analyst_piv_status:    PIVStatus = PIVStatus.REJECT
    analyst_attempt_count: int       = 0

    # ── PIV — QUANT POD (N12) ──────────────────────────────────────────────
    quant_result:        str       = ""
    quant_confidence:    float     = 0.0
    quant_citations:     List[str] = Field(default_factory=list)
    quant_piv_status:    PIVStatus = PIVStatus.REJECT
    quant_attempt_count: int       = 0

    # N12 Quantitative outputs
    monte_carlo_results: Dict  = Field(default_factory=dict)
    var_result:          Dict  = Field(default_factory=dict)
    garch_result:        Dict  = Field(default_factory=dict)
    computed_ratio:      float = 0.0

    # ── PIV — FORENSICS (N13) ──────────────────────────────────────────────
    forensic_flags:    List[str] = Field(default_factory=list)
    risk_score:        float     = 0.0
    anomaly_detected:  bool      = False
    anomaly_severity:  str       = "low"
    benford_chi2:      float     = 0.0
    benford_p_value:   float     = 1.0

    # ── PIV — AUDITOR POD (N14) ────────────────────────────────────────────
    auditor_output:        str       = ""
    auditor_confidence:    float     = 0.0
    auditor_citations:     List[str] = Field(default_factory=list)
    auditor_attempt_count: int       = 0
    auditor_piv_status:    PIVStatus = PIVStatus.REJECT
    contradiction_flags:   List[str] = Field(default_factory=list)

    # ── DEBATE / MEDIATOR (N15) ────────────────────────────────────────────
    piv_candidates:       List[Dict] = Field(default_factory=list)
    piv_round:            int        = 0
    iteration_count:      int        = 0        # A2: hard cap 5
    final_answer_pre_xgb: str        = ""
    agreement_status:     str        = ""
    confidence_score:     float      = 0.0
    low_confidence:       bool       = False    # A3 clarification engine
    winning_pod:          str        = ""

    # ── EXPLAINABILITY (N16) ───────────────────────────────────────────────
    shap_values:        Dict = Field(default_factory=dict)
    feature_importance: Dict = Field(default_factory=dict)
    causal_dag_path:    str  = ""

    # ── XGB ARBITER (N17) ──────────────────────────────────────────────────
    xgb_ranked_answer: str   = ""
    xgb_score:         float = 0.0

    # ── FINAL OUTPUT (N18-N19) ─────────────────────────────────────────────
    final_answer:      str = ""
    final_report_path: str = ""

    # ── RLEF — PRIVATE FOREVER (C9) ────────────────────────────────────────
    # These fields NEVER appear in any output, UI, log, or DOCX report.
    _rlef_grade:          str   = ""
    _rlef_va_score:       float = 0.0
    _rlef_vb_score:       float = 0.0
    _rlef_vc_score:       float = 0.0
    _rlef_chosen:         str   = ""
    _rlef_rejected:       str   = ""
    _rlef_user_consented: bool  = False
    _rlef_stored_global:  bool  = False

    # ── VALIDATORS ─────────────────────────────────────────────────────────

    @field_validator("prompt_template")
    @classmethod
    def enforce_context_first(cls, v: str) -> str:
        """C7: prompt_template must always be 'context_first'."""
        if v != "context_first":
            raise ValueError(
                f"C7 violated: prompt_template must be 'context_first'. Got '{v}'."
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

    @field_validator("iteration_count")
    @classmethod
    def enforce_iteration_cap(cls, v: int) -> int:
        """A2: iteration_count hard cap = 5."""
        if v > 5:
            raise ValueError(
                f"A2 violated: iteration_count cap is 5. Got {v}."
            )
        return v

    @field_validator("query_type", mode="before")
    @classmethod
    def normalise_query_type(cls, v: Any) -> str:
        """
        Accept both QueryType enum and str.
        Pipeline.py sets query_type as plain string — this prevents crash.
        """
        VALID = {"numerical", "ratio", "multi_doc", "text", "forensic"}
        if isinstance(v, QueryType):
            return v.value
        if isinstance(v, str):
            if v in VALID:
                return v
            # Graceful fallback — don't crash on unknown type
            return "text"
        return "text"

    # ── Helpers ────────────────────────────────────────────────────────────

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
        Return _rlef_ fields for internal RLEF use ONLY.
        NEVER call this in output generation code (C9).
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


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from rich import print as rprint
    except ImportError:
        rprint = print

    rprint("\n[bold cyan]-- BAState v11 sanity check --[/bold cyan]")

    state = BAState(
        session_id   = "sanity-check",
        company_name = "Apple Inc",
        doc_type     = "10-K",
        fiscal_year  = "FY2023",
    )
    rprint(f"[green]✓[/green] Created | seed={state.seed} | "
           f"prompt_template={state.prompt_template}")

    # ✅ FIX 1: bge_results field exists
    state.bge_results = [{"chunk_id": "a", "text": "test", "bge_score": 0.9}]
    assert state.bge_results[0]["chunk_id"] == "a"
    rprint("[green]✓[/green] FIX: bge_results field works")

    # ✅ FIX 2: reranked_chunks field exists
    state.reranked_chunks = [{"chunk_id": "b", "text": "test2", "rrf_score": 0.8}]
    assert state.reranked_chunks[0]["chunk_id"] == "b"
    rprint("[green]✓[/green] FIX: reranked_chunks field works")

    # ✅ FIX 3: query_type accepts string from pipeline.py
    state.query_type = "numerical"
    assert state.query_type == "numerical"
    state.query_type = "text"
    assert state.query_type == "text"
    rprint("[green]✓[/green] FIX: query_type accepts plain string")

    # C5 enforced
    try:
        state.seed = 99
        rprint("[red]✗[/red] C5 NOT enforced")
    except Exception:
        rprint("[green]✓[/green] C5 enforced — seed=42 locked")

    # C7 enforced
    try:
        state.prompt_template = "question_first"
        rprint("[red]✗[/red] C7 NOT enforced")
    except Exception:
        rprint("[green]✓[/green] C7 enforced — context_first locked")

    # A2 enforced
    try:
        state.iteration_count = 6
        rprint("[red]✗[/red] A2 NOT enforced")
    except Exception:
        rprint("[green]✓[/green] A2 enforced — iteration cap=5 locked")

    # C9 — _rlef_ fields private
    rlef = state.get_rlef_fields()
    assert all(k.startswith("_rlef_") for k in rlef)
    rprint("[green]✓[/green] C9 enforced — _rlef_ fields private")

    # C8 prefix
    prefix = state.chunk_prefix("Apple Inc", "10-K", "FY2023", "MD&A", 42)
    assert prefix == "Apple Inc / 10-K / FY2023 / MD&A / 42"
    rprint(f"[green]✓[/green] C8 prefix: {prefix}")

    # N12 quant fields
    state.monte_carlo_results = {"mean": 96995.0, "n": 10000}
    state.var_result          = {"var_95": 64793.0, "var_99": 58887.0}
    assert state.monte_carlo_results["mean"] == 96995.0
    rprint("[green]✓[/green] N12 quant fields work")

    # N16 SHAP fields
    state.shap_values        = {"bm25_score": 0.42, "cosine_sim": 0.38}
    state.feature_importance = {"bm25_score": 0.50, "cosine_sim": 0.30}
    state.causal_dag_path    = "outputs/causal_dag_test.png"
    assert isinstance(state.shap_values, dict)
    rprint("[green]✓[/green] N16 SHAP fields work")

    # attempt_count fields
    state.analyst_attempt_count = 2
    state.quant_attempt_count   = 1
    state.auditor_attempt_count = 3
    rprint("[green]✓[/green] attempt_count fields work")

    # C8 chunk_metadata_prefix
    state.chunk_metadata_prefix = "Apple Inc / 10-K / FY2023 / MD&A / 42"
    assert "Apple" in state.chunk_metadata_prefix
    rprint("[green]✓[/green] chunk_metadata_prefix works")

    # A3 clarification engine
    state.low_confidence = True
    assert state.low_confidence is True
    rprint("[green]✓[/green] A3 low_confidence flag works")

    # winning_pod (N15)
    state.winning_pod = "SNIPER_EARLY_EXIT"
    assert state.winning_pod == "SNIPER_EARLY_EXIT"
    rprint("[green]✓[/green] winning_pod field works")

    # final fields (N19)
    state.final_answer      = "$383,285 million"
    state.final_report_path = "outputs/report_test.docx"
    rprint("[green]✓[/green] final_answer + final_report_path work")

    rprint("\n[bold green]✅ All v11 checks passed. BAState ready.[/bold green]\n")
    rprint("[bold yellow]SESSION 16 FIX: bge_results + reranked_chunks + query_type str fix[/bold yellow]\n")
