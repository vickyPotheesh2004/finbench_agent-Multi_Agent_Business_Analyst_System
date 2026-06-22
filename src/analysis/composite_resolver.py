"""
src/analysis/composite_resolver.py
N20 — Composite Resolver (the "20th node" / JEE-method orchestrator)

The single entry point for OFFLINE answering of complex questions that
SNIPER + formula_router cannot solve alone.

Routes a question to the right sub-resolver:

  ┌───────────────────────────────────────────────────────────────┐
  │  question  ─┬─►  detect type ─┬─► numeric          (sniper)   │
  │             │                 ├─► ratio formula    (router)   │
  │             │                 ├─► yes/no decision  (decision) │
  │             │                 ├─► what drove X     (narrative)│
  │             │                 └─► which segment    (narrative)│
  └───────────────────────────────────────────────────────────────┘

All paths are 100 % deterministic, ZERO LLM, full audit trail.

Integration: called from pipeline.py BEFORE the SNIPER_ONLY fallback
exits. If composite returns answered=True we use that and skip the
sniper-only retrieval miss path entirely.

Constraints satisfied:
  C1, C2 ($0, local)  |  C5 (no randomness)  |  C9 (no _rlef_ writes)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.analysis.decision_engine import (
    DecisionAnswer,
    detect_decision_pattern,
    run_decision_engine,
)
from src.analysis.narrative_extractor import (
    NarrativeAnswer,
    detect_causal_question,
    detect_segment_question,
    extract_drivers,
    extract_segment_answer,
)

logger = logging.getLogger(__name__)

# Optional lib imports
try:
    from src.utils.lib_bridge import (
        classify_question_safe,
        verify_value_safe,
        verify_final_answer_safe,
    )
    _HAS_LIB_BRIDGE = True
except Exception:
    classify_question_safe = None
    verify_value_safe = None
    verify_final_answer_safe = None
    _HAS_LIB_BRIDGE = False

# Optional: question_lib (the JEE side-word decomposer)
try:
    from question_lib import answer_question as _ql_answer_question
    from question_lib import parse_question as _ql_parse_question
    _HAS_QUESTION_LIB = True
except Exception:
    _ql_answer_question = None
    _ql_parse_question = None
    _HAS_QUESTION_LIB = False

# Question simplifier (2026-06-13): rewrites twisted FinanceBench phrasing into
# clean canonical form BEFORE routing/classification, so the deterministic
# resolvers and (later) the LLM see the simple version. Safe: returns the
# original untouched if it can't simplify confidently.
try:
    from src.analysis.question_simplifier import simplify_question as _simplify
    _HAS_SIMPLIFIER = True
except Exception:
    _simplify = None
    _HAS_SIMPLIFIER = False


# ─────────────────────────────────────────────────────────────────────────────
# Composite answer dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompositeAnswer:
    """Unified return type for any sub-resolver."""
    answered: bool
    final_answer: str = ""
    winning_pod: str = ""               # which sub-resolver won
    confidence: float = 0.0
    classification: str = ""            # for decision questions
    drivers: List[str] = field(default_factory=list)   # for narrative questions
    audit_trail: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Question classifier — pre-flight routing
# ─────────────────────────────────────────────────────────────────────────────

# Narrative / qualitative question detector (2026-06-21). Returns True when a
# question wants a TEXT answer (industry, diversification, customers, legal
# proceedings, acquisitions list, yes/no rationale) rather than a single
# number. Used to STOP question_lib from emitting a stray dollar value that
# scores 0 and blocks the LLM. Deliberately conservative: a question that
# names a concrete numeric metric (revenue, margin, ratio, DPO, turnover,
# capex, EPS...) is NOT treated as narrative even if it also has narrative
# words, so we never abstain on a real numeric question.
_NUMERIC_METRIC_WORDS = (
    "revenue", "net sales", "net income", "margin", "ratio", "turnover",
    "dpo", "dso", "dio", "days payable", "days sales", "capex",
    "capital expenditure", "eps", "earnings per share", "cogs",
    "cost of", "assets", "liabilities", "equity", "cash flow",
    "ebitda", "dividend", "inventory", "depreciation", "how much",
    "how many times", "what is the", "what was the", "yoy",
    "year-over-year", "year over year", "quick ratio", "current ratio",
    "return on", "working capital", "tax rate", "effective tax",
)
_NARRATIVE_CUES = (
    "what industry", "what is the nature", "what are the major",
    "major acquisitions", "what acquisitions", "diversification",
    "diversified", "product categories", "primary customers",
    "key customers", "legal proceedings", "lawsuit", "litigation",
    "key agenda", "agenda of", "what drove", "what caused", "why did",
    "what is the purpose", "nature & purpose", "nature and purpose",
    "restructuring", "primarily operate", "business segment",
    "competitive", "strategy", "outlook", "guidance", "risk factor",
    "who is", "is the ceo", "new ceo", "spin off", "spin-off",
    "spinning off", "real change in sales", "real growth",
    "adjusted non gaap", "adjusted non-gaap", "non gaap ebitda",
    "non-gaap ebitda", "adj. ebitda", "adjusted ebitda",
    "real change", "organic change", "organic growth",
)


def _is_narrative_question(question: str) -> bool:
    """True for clearly qualitative questions that must NOT be answered with a
    single extracted number. Conservative: a concrete numeric metric word
    overrides the narrative cue so real numeric questions still compute."""
    q = (question or "").lower()
    if not q:
        return False
    has_narr = any(cue in q for cue in _NARRATIVE_CUES)
    if not has_narr:
        return False
    # If the question ALSO names a concrete numeric metric, it's a real
    # numeric question (e.g. "what drove the change in revenue" still wants a
    # number sometimes) -> let it compute. Only abstain when it's PURELY
    # narrative. Exception: 'real change in sales' / 'real growth' are the
    # FinanceBench narrative-comparison traps -> always narrative.
    if "real change in sales" in q or "real growth" in q:
        return True
    if any(w in q for w in _NUMERIC_METRIC_WORDS):
        return False
    return True


def classify_question(question: str) -> str:
    """
    Return one of:
      'decision'   - Is X capital-intensive? Is liquidity healthy?
      'causal'     - What drove margin change? Why did revenue decline?
      'segment'    - Which segment had X?
      'other'      - fall through to existing numeric paths
    """
    if not question:
        return "other"

    # Decision patterns FIRST (most specific)
    if detect_decision_pattern(question) is not None:
        return "decision"

    # Segment questions before causal (more specific phrasing)
    if detect_segment_question(question):
        return "segment"

    # Causal "what drove" / "why did"
    if detect_causal_question(question):
        return "causal"

    # Try pattern_lib if we have it (informational)
    if _HAS_LIB_BRIDGE and classify_question_safe is not None:
        try:
            pat_res = classify_question_safe(question)
            if pat_res:
                qt = pat_res.get("question_type", "")
                # pattern_lib has its own taxonomy; we only use it as a hint
                logger.debug("[composite] pattern_lib type: %s", qt)
        except Exception:
            pass

    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_composite_resolver(state) -> CompositeAnswer:
    """
    Main entry point. Reads from BAState (duck-typed) and returns a
    CompositeAnswer. Never raises.
    """
    try:
        question = getattr(state, "query", "") or ""
        if not question:
            return CompositeAnswer(answered=False)

        # 2026-06-13: simplify twisted phrasing BEFORE routing. We keep the
        # ORIGINAL for citation/audit but route on the simplified text. If the
        # simplifier changed nothing, this is a no-op.
        original_question = question
        if _HAS_SIMPLIFIER and _simplify is not None:
            try:
                sq = _simplify(question)
                if sq and sq.simplified and sq.changed:
                    logger.info("[composite] simplified: %r -> %r",
                                original_question[:60], sq.simplified[:60])
                    question = sq.simplified
            except Exception:
                logger.debug("[composite] simplifier failed", exc_info=True)

        company     = getattr(state, "company_name", "") or ""
        fiscal_year = getattr(state, "fiscal_year",  "") or ""
        doc_type    = getattr(state, "doc_type",     "") or ""
        raw_text    = getattr(state, "raw_text",     "") or ""
        cells       = getattr(state, "table_cells",  None) or []
        structured_tables = getattr(state, "structured_tables", None) or []
        bm25_results = getattr(state, "bm25_results", None) or []

        qtype = classify_question(question)
        logger.info("[composite] Question type: %s", qtype)

        # ── DECISION (yes/no, classification) ──────────────────────────
        if qtype == "decision":
            dec: DecisionAnswer = run_decision_engine(
                question=question,
                cells=cells,
                raw_text=raw_text,
                company=company,
                fiscal_year=fiscal_year,
                doc_type=doc_type,
            )
            if dec.answered:
                # FIX-v5: Hard-block when verify_lib says abstain (not just warn).
                if _HAS_LIB_BRIDGE and verify_value_safe is not None and dec.metric_value is not None:
                    try:
                        ok, abstain, reasons = verify_value_safe(
                            metric=dec.rule_id,
                            value=dec.metric_value,
                            confidence=dec.confidence,
                        )
                        if abstain or not ok:
                            logger.warning(
                                "[composite] BLOCKING decision answer: verify_lib abstain | %s",
                                "; ".join(reasons)[:200],
                            )
                            return CompositeAnswer(
                                answered=False,
                                audit_trail={
                                    "reason":   "verify_lib_abstain",
                                    "value":    dec.metric_value,
                                    "reasons":  reasons,
                                    "original": dec.output[:200],
                                },
                            )
                    except Exception:
                        logger.debug("[composite] verify_lib check failed",
                                     exc_info=True)
                return CompositeAnswer(
                    answered=True,
                    final_answer=dec.output,
                    winning_pod="N20_DECISION",
                    confidence=dec.confidence,
                    classification=dec.classification,
                    audit_trail=dec.audit_trail,
                )

        # ── CAUSAL ("what drove X?") ───────────────────────────────────
        if qtype == "causal":
            nar: NarrativeAnswer = extract_drivers(
                question=question,
                raw_text=raw_text,
                bm25_results=bm25_results,
                company=company,
                fiscal_year=fiscal_year,
                doc_type=doc_type,
            )
            if nar.answered:
                return CompositeAnswer(
                    answered=True,
                    final_answer=nar.output,
                    winning_pod="N20_NARRATIVE",
                    confidence=nar.confidence,
                    drivers=nar.drivers,
                    audit_trail=nar.audit_trail,
                )

        # ── SEGMENT ("which segment ...") ──────────────────────────────
        if qtype == "segment":
            seg: NarrativeAnswer = extract_segment_answer(
                question=question,
                raw_text=raw_text,
                bm25_results=bm25_results,
                company=company,
                fiscal_year=fiscal_year,
                doc_type=doc_type,
                cells=cells,   # MOVE-5: enables segment-growth-from-tables fallback
            )
            if seg.answered:
                return CompositeAnswer(
                    answered=True,
                    final_answer=seg.output,
                    winning_pod="N20_SEGMENT",
                    confidence=seg.confidence,
                    drivers=seg.drivers,
                    audit_trail=seg.audit_trail,
                )

        # ── Fall through: not a composite question (numeric / "other") ─
        # ── Try question_lib as a LAST RESORT for COMPUTE / PROJECT  ─
        # NOTE: question_lib needs the ORIGINAL question text — it parses
        # exact periods ("FY2017 - FY2019"), metric names and operations that
        # the simplifier may have collapsed. Routing above uses the simplified
        # text; computation here uses the original.
        #
        # NARRATIVE GUARD (2026-06-21): the 150-run showed question_lib
        # answering QUALITATIVE questions with a stray dollar number
        # ("Does Boeing have product diversification?" -> $66,608M;
        # "What industry does Amcor operate in?" -> a number). Those are
        # reading-comprehension questions with a TEXT gold answer; emitting a
        # number scores 0 AND blocks the LLM. If the question is clearly
        # narrative/qualitative, ABSTAIN here so it falls through to the LLM.
        if _is_narrative_question(original_question):
            logger.info("[composite] narrative question -> abstain from "
                        "question_lib numeric path: %r",
                        original_question[:70])
            return CompositeAnswer(
                answered=False,
                audit_trail={"qtype": qtype,
                             "reason": "narrative_question_abstain"},
            )

        if _HAS_QUESTION_LIB and _ql_answer_question is not None:
            try:
                ql_res = _ql_answer_question(
                    question=original_question,
                    cells=cells,
                    raw_text=raw_text,
                    company=company,
                    fiscal_year=fiscal_year,
                    doc_type=doc_type,
                    structured_tables=structured_tables,
                )
                if ql_res and ql_res.answered:
                    logger.info("[composite] question_lib HIT | answer=%s",
                                 ql_res.final_answer[:140])
                    return CompositeAnswer(
                        answered=True,
                        final_answer=ql_res.final_answer,
                        winning_pod="N20_QUESTION_LIB",
                        confidence=ql_res.confidence,
                        audit_trail=ql_res.audit_trail,
                    )
            except Exception:
                logger.exception("[composite] question_lib failed")

        return CompositeAnswer(
            answered=False,
            audit_trail={"qtype": qtype, "reason": "no sub-resolver matched"},
        )

    except Exception:
        logger.exception("[composite] run_composite_resolver crashed")
        return CompositeAnswer(answered=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-style wrapper — mutates state in place, returns state
# ─────────────────────────────────────────────────────────────────────────────

def run_composite_node(state):
    """
    Pipeline-node wrapper: if composite resolver answers, set state fields
    and return state. Otherwise leave state untouched.

    Caller should check `state.composite_hit` (we set it on success) to
    decide whether to short-circuit the rest of the pipeline.
    """
    ans = run_composite_resolver(state)

    # Always record audit info on private attribute (NOT _rlef_)
    try:
        setattr(state, "composite_audit", ans.audit_trail)
    except Exception:
        pass

    if ans.answered:
        try:
            state.final_answer         = ans.final_answer
            state.final_answer_pre_xgb = ans.final_answer
            state.confidence_score     = ans.confidence
            state.winning_pod          = ans.winning_pod
            # Mark hit so pipeline can short-circuit SNIPER_ONLY fallback
            setattr(state, "composite_hit", True)
            logger.info(
                "[composite] HIT  | pod=%s | conf=%.2f | answer=%s",
                ans.winning_pod,
                ans.confidence,
                ans.final_answer[:120],
            )
        except Exception:
            logger.exception("[composite] failed to write state fields")
    else:
        try:
            setattr(state, "composite_hit", False)
        except Exception:
            pass

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    class _FakeState:
        def __init__(self, q, raw, company, fy, dt):
            self.query        = q
            self.raw_text     = raw
            self.company_name = company
            self.fiscal_year  = fy
            self.doc_type     = dt
            self.table_cells  = []
            self.bm25_results = []

    cases = [
        ("Is 3M a capital-intensive business based on FY2022 data?",
         "Capital expenditures for FY2022 totaled $1,749 million for property, "
         "plant and equipment. Net sales for the year were $34,229 million.",
         "3M", "FY2022", "10-K"),
        ("What drove operating margin change as of FY2022 for 3M?",
         "The decrease in operating margin was primarily driven by significant "
         "litigation charges related to Combat Arms earplug matters and Aearo "
         "Technologies, as well as the exit from PFAS manufacturing. Results "
         "were also impacted by divestiture activity and the exit from Russia.",
         "3M", "FY2022", "10-K"),
        ("If we exclude the impact of M&A, which segment has dragged down "
         "3M's overall growth in 2022?",
         "Safety and Industrial revenue grew 4.5% organically. "
         "Transportation and Electronics revenue grew 2.1%. "
         "Health Care revenue grew 8.3%. "
         "Consumer revenue declined by 0.9% organically as demand softened.",
         "3M", "FY2022", "10-K"),
    ]

    print("\n══════ N20 Composite Resolver self-test ══════\n")
    for i, (q, raw, c, fy, dt) in enumerate(cases, 1):
        print(f"\n── Case {i} ─────────────────────────────────────────")
        print(f"Q: {q}")
        s = _FakeState(q, raw, c, fy, dt)
        s = run_composite_node(s)
        print(f"answered:  {getattr(s, 'composite_hit', False)}")
        print(f"pod:       {getattr(s, 'winning_pod', '')}")
        print(f"conf:      {getattr(s, 'confidence_score', 0.0):.2f}")
        print(f"answer:    {getattr(s, 'final_answer', '')[:300]}")

    print("\n══════ N20 ready for pipeline integration ══════\n")
