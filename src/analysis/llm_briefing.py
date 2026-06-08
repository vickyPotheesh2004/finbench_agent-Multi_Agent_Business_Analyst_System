"""
src/analysis/llm_briefing.py
─────────────────────────────────────────────────────────────────────────────
LLM Briefing Assembler — "give Llama 3.1 8B a research analyst's briefing"

THE IDEA (babe's insight):
    Small models (8B) hallucinate on financial narrative because they have
    to do EVERYTHING at once: understand the question, find the relevant
    text, AND reason. We split that load:

        question_lib.classify_intent()  → tells the LLM WHAT KIND of question
        question_lib.parse_question()   → tells the LLM the subject/period
        narrative_extractor             → pre-finds candidate driver phrases
        extract_lib                     → gives the LLM the exact numbers
        retrieval (BM25/BGE)            → gives the LLM the relevant paragraphs

    We package all of that into a STRUCTURED BRIEFING and hand it to the LLM.
    The LLM's only job becomes: "read this briefing and write the answer."
    That is a MUCH easier task for an 8B model → higher accuracy.

This is the proven "decompose + tool-augment" pattern (ReAct / RAG-fusion).
It does NOT replace the LLM — it makes the LLM's job small and well-defined.

100% local. $0. Uses ONLY existing libs. No new dependencies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional lib imports (graceful) ──────────────────────────────────────────

try:
    from question_lib.intent_classifier import classify_intent as _classify_intent
    from question_lib.models import Intent
    _HAS_INTENT = True
except Exception:
    _classify_intent = None
    Intent = None
    _HAS_INTENT = False

try:
    from question_lib import parse_question as _parse_question
    _HAS_PARSE = True
except Exception:
    _parse_question = None
    _HAS_PARSE = False

try:
    from src.analysis.narrative_extractor import (
        extract_drivers,
        extract_segment_answer,
        detect_causal_question,
        detect_segment_question,
        detect_subject,
    )
    _HAS_NARRATIVE = True
except Exception:
    _HAS_NARRATIVE = False

try:
    from extract_lib.resolver import resolve_metric as _resolve_metric
    _HAS_EXTRACT = True
except Exception:
    _resolve_metric = None
    _HAS_EXTRACT = False


# ─────────────────────────────────────────────────────────────────────────────
# Intent → guidance text that primes the LLM
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_GUIDANCE: Dict[str, str] = {
    "extract": (
        "This is a VALUE-EXTRACTION question. The answer is a single number "
        "that appears verbatim in the context. Find it, state it with units "
        "and fiscal year. Do NOT compute or estimate."
    ),
    "compute": (
        "This is a CALCULATION question. Identify the formula, pull each input "
        "value from the context, show the arithmetic, then state the result."
    ),
    "decide": (
        "This is a YES/NO or CLASSIFICATION question. Compute the relevant "
        "metric, compare it against a sensible threshold, then give a clear "
        "verdict with the supporting number."
    ),
    "compare": (
        "This is a COMPARISON question. Find each item's value, then state "
        "which is higher/lower and by how much."
    ),
    "narrate": (
        "This is a NARRATIVE / CAUSAL question (\"what drove X\", \"which "
        "segment\", \"why did Y\"). The answer is a SHORT LIST OF DRIVERS or a "
        "specific named segment, taken ONLY from the management discussion in "
        "the context. List the named causes (e.g. litigation, divestitures, "
        "FX, pricing, volume). Do NOT invent drivers not present in the text."
    ),
    "project": (
        "This is a PROJECTION question. Take the stated base value, apply the "
        "stated growth/assumption, show the arithmetic, state the projected "
        "value. Mark it clearly as an estimate."
    ),
    "unknown": (
        "Read the context carefully and answer only from what it states."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# The briefing dataclass-ish dict
# ─────────────────────────────────────────────────────────────────────────────

def build_briefing(
    question: str,
    raw_text: str = "",
    cells: Optional[List[Dict]] = None,
    bm25_results: Optional[List[Dict]] = None,
    company: str = "",
    fiscal_year: str = "",
    doc_type: str = "",
) -> Dict[str, Any]:
    """
    Run all the deterministic libs and collect their findings into a
    structured briefing the LLM can consume.

    Returns a dict with keys:
        intent, intent_confidence, polarity, guidance,
        subject, period,
        candidate_drivers   (for narrative questions)
        candidate_segment   (for segment questions)
        known_numbers       (metric -> value, from extract_lib)
        notes
    """
    cells = cells or []
    bm25_results = bm25_results or []

    briefing: Dict[str, Any] = {
        "question":          question,
        "intent":            "unknown",
        "intent_confidence": 0.0,
        "polarity":          "neutral",
        "guidance":          _INTENT_GUIDANCE["unknown"],
        "subject":           "",
        "period":            fiscal_year,
        "candidate_drivers": [],
        "candidate_segment": "",
        "known_numbers":     {},
        "notes":             [],
    }

    # ── 1. Intent classification (your existing classifier) ──────────────
    if _HAS_INTENT and _classify_intent is not None:
        try:
            intent, polarity, conf = _classify_intent(question)
            iv = intent.value if hasattr(intent, "value") else str(intent)
            pv = polarity.value if hasattr(polarity, "value") else str(polarity)
            briefing["intent"]            = iv
            briefing["intent_confidence"] = round(float(conf), 2)
            briefing["polarity"]          = pv
            briefing["guidance"]          = _INTENT_GUIDANCE.get(iv, _INTENT_GUIDANCE["unknown"])
        except Exception:
            logger.debug("[briefing] intent classify failed", exc_info=True)

    # ── 2. Subject + period (parse_question) ─────────────────────────────
    if _HAS_PARSE and _parse_question is not None:
        try:
            plan = _parse_question(question)
            if plan.subject:
                briefing["subject"] = plan.subject.display_name or plan.subject.metric_id
            if plan.periods:
                p0 = plan.periods[0]
                briefing["period"] = p0.fiscal_year or p0.raw or fiscal_year
        except Exception:
            logger.debug("[briefing] parse_question failed", exc_info=True)

    # Fallback subject detection
    if not briefing["subject"] and _HAS_NARRATIVE:
        try:
            subj = detect_subject(question)
            if subj:
                briefing["subject"] = subj
        except Exception:
            pass

    # ── 3. Narrative pre-mining (for narrate questions) ──────────────────
    if _HAS_NARRATIVE and briefing["intent"] == "narrate":
        try:
            if detect_causal_question(question):
                nar = extract_drivers(
                    question, raw_text, bm25_results,
                    company, fiscal_year, doc_type,
                )
                if nar.answered and nar.drivers:
                    briefing["candidate_drivers"] = nar.drivers
                    briefing["notes"].append(
                        f"Deterministic driver-miner found candidates: "
                        f"{', '.join(nar.drivers)}. Verify each against the context."
                    )
            if detect_segment_question(question):
                seg = extract_segment_answer(
                    question, raw_text, bm25_results,
                    company, fiscal_year, doc_type,
                )
                if seg.answered and seg.drivers:
                    briefing["candidate_segment"] = seg.drivers[0]
                    briefing["notes"].append(
                        f"Deterministic segment-finder suggests: "
                        f"{seg.drivers[0]}. Verify against the context."
                    )
        except Exception:
            logger.debug("[briefing] narrative pre-mining failed", exc_info=True)

    # ── 4. Known numbers (extract_lib) — give the LLM hard facts ─────────
    if _HAS_EXTRACT and _resolve_metric is not None and cells:
        # Pull a small set of common metrics that anchor most questions
        anchor_metrics = _anchor_metrics_for(briefing["subject"], briefing["intent"])
        for mid in anchor_metrics:
            try:
                r = _resolve_metric(mid, cells, briefing["period"] or "")
                if r and getattr(r, "valid", False) and r.value is not None:
                    briefing["known_numbers"][mid] = {
                        "value":      r.value,
                        "row_header": getattr(r, "row_header", ""),
                        "page":       getattr(r, "page", None),
                    }
            except Exception:
                continue

    return briefing


def _anchor_metrics_for(subject: str, intent: str) -> List[str]:
    """Pick a few metrics worth pre-extracting to anchor the LLM."""
    s = (subject or "").lower()
    base = []
    if "margin" in s or intent in ("compute", "narrate", "decide"):
        base += ["revenue", "operating_income", "net_income", "cogs", "gross_profit"]
    if "capital" in s or "capex" in s or "intensive" in s:
        base += ["capex", "revenue"]
    if "segment" in s:
        base += ["revenue"]
    if not base:
        # generic anchors
        base = ["revenue", "net_income", "total_assets"]
    # dedupe preserving order
    seen = set()
    out = []
    for m in base:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Render the briefing into a prompt block the LLM reads BEFORE the question
# ─────────────────────────────────────────────────────────────────────────────

def render_briefing_block(briefing: Dict[str, Any]) -> str:
    """
    Turn the briefing dict into a compact text block for the LLM prompt.
    Placed BEFORE the question (C7-compatible — context/briefing first).
    """
    lines: List[str] = []
    lines.append("══ ANALYST BRIEFING (pre-computed by deterministic tools) ══")
    lines.append(f"Question type: {briefing['intent'].upper()} "
                 f"(confidence {briefing['intent_confidence']})")
    lines.append(f"How to answer this type: {briefing['guidance']}")

    if briefing.get("subject"):
        lines.append(f"Subject metric: {briefing['subject']}")
    if briefing.get("period"):
        lines.append(f"Fiscal period: {briefing['period']}")
    if briefing.get("polarity") and briefing["polarity"] != "neutral":
        lines.append(f"Question framing: {briefing['polarity']}")

    # Hard numbers the tools already found — anchors the LLM, kills hallucination
    known = briefing.get("known_numbers") or {}
    if known:
        lines.append("\nVerified figures (from table extraction — TRUST THESE):")
        for mid, info in known.items():
            val = info.get("value")
            rh  = info.get("row_header", "")
            pg  = info.get("page")
            tag = f" [row: {rh}]" if rh else ""
            pgtag = f" (p.{pg})" if pg is not None else ""
            lines.append(f"  • {mid} = {val}{tag}{pgtag}")

    # Candidate drivers (narrative)
    if briefing.get("candidate_drivers"):
        lines.append("\nCandidate drivers found in the text "
                     "(verify each before including):")
        for d in briefing["candidate_drivers"]:
            lines.append(f"  • {d}")

    if briefing.get("candidate_segment"):
        lines.append(f"\nCandidate segment (verify against context): "
                     f"{briefing['candidate_segment']}")

    if briefing.get("notes"):
        lines.append("\nTool notes:")
        for n in briefing["notes"]:
            lines.append(f"  • {n}")

    lines.append("══ END BRIEFING ══")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    q = "What drove operating margin change as of FY2022 for 3M?"
    raw = (
        "Operating income margin was 19.4% compared to 22.2% last year. "
        "The decrease in operating margin was primarily driven by significant "
        "litigation charges related to Combat Arms earplug matters, "
        "as well as the exit from PFAS manufacturing. Results were also "
        "impacted by divestiture activity and the exit from Russia."
    )
    cells = [
        {"row_header": "net sales", "col_header": "2022", "value": "34229"},
        {"row_header": "operating income", "col_header": "2022", "value": "6539"},
    ]

    b = build_briefing(q, raw_text=raw, cells=cells,
                        company="3M", fiscal_year="FY2022", doc_type="10-K")
    print("\n── BRIEFING DICT ──")
    for k, v in b.items():
        print(f"  {k}: {v}")
    print("\n── RENDERED BLOCK ──")
    print(render_briefing_block(b))
