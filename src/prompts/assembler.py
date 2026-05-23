"""
N10 Prompt Assembler — Context-First Prompt Construction
PDR-BAAAI-001 · Rev 1.0 · Node N10

Purpose:
    Build the final LLM prompt from retrieved chunks (retrieval_stage_2).
    5 Jinja2 templates: numerical, ratio, multi_doc, text, forensic.
    retrieved_context ALWAYS appears BEFORE the question — C7 enforced.
    Injects company name, fiscal year, units, and query type context.

Constraints satisfied:
    C1  $0 cost — Jinja2 is free
    C2  100% local — zero network calls
    C5  seed=42 — no randomness at this node
    C7  Context BEFORE question — enforced by Pydantic validator +
        CI/CD gate + runtime assertion in every template render
    C8  Chunk metadata prefix present in all retrieved context
    C9  No _rlef_ fields in any prompt output
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Jinja2 lazy import ────────────────────────────────────────────────────────

_jinja2 = None

def _get_jinja2():
    global _jinja2
    if _jinja2 is None:
        import jinja2
        _jinja2 = jinja2
    return _jinja2


# ── Constants ─────────────────────────────────────────────────────────────────

QUERY_TYPES     = ["numerical", "ratio", "multi_doc", "text", "forensic"]
MAX_CHUNKS      = 5       # maximum chunks to include in context
MAX_CHUNK_CHARS = 1500    # maximum characters per chunk (prevents token overflow)

# C7 enforcement marker — every prompt must contain this before the question
_CONTEXT_MARKER  = "RETRIEVED CONTEXT"
_QUESTION_MARKER = "QUESTION"


# ── Jinja2 Templates — 5 query type variants ──────────────────────────────────
# C7: retrieved_context ALWAYS first — never move it below the question
# Each template verified: CONTEXT section appears before QUESTION section

TEMPLATES: Dict[str, str] = {

    # ── numerical — direct value extraction ───────────────────────────────────
    "numerical": """You are a precise financial analyst extracting exact numerical values.
Your ONLY source of truth is the retrieved context below.
Never use training knowledge. Never guess.

RETRIEVED CONTEXT (your ONLY source — never use prior knowledge):
{{ retrieved_context }}

COMPANY: {{ company_name }}
FISCAL YEAR: {{ fiscal_year }}
QUERY TYPE: numerical

QUESTION: {{ question }}

STRICT RULES:
RULE 1: Answer ONLY from the retrieved context above.
RULE 2: If the answer is not in context output RETRIEVAL_MISS with exact info needed.
RULE 3: Cite every number: [SECTION_NAME / PAGE_NUM: value]
RULE 4: State units explicitly: millions, billions, or percentage.
RULE 5: State fiscal year for every figure cited.
RULE 6: Never guess. Never use training memory.

Output format:
ANSWER: [exact value with units and fiscal year]
COMPUTATION: N/A
CONFIDENCE: [0.0-1.0] because [brief justification]
CITATIONS: [list every section/page reference used]
""",

    # ── ratio — formula computation ───────────────────────────────────────────
    "ratio": """You are a precise financial analyst computing ratios and metrics.
Your ONLY source of truth is the retrieved context below.
Never use training knowledge. Never guess.

RETRIEVED CONTEXT (your ONLY source — never use prior knowledge):
{{ retrieved_context }}

COMPANY: {{ company_name }}
FISCAL YEAR: {{ fiscal_year }}
QUERY TYPE: ratio

QUESTION: {{ question }}

STRICT RULES:
RULE 1: Answer ONLY from the retrieved context above.
RULE 2: If inputs are not in context output RETRIEVAL_MISS with exact info needed.
RULE 3: Show the complete formula before computing.
RULE 4: Show all inputs with their cited values.
RULE 5: State units explicitly for every figure.
RULE 6: State fiscal year for every figure cited.

Output format:
ANSWER: [ratio or percentage result]
COMPUTATION: [formula: numerator / denominator = result, show all inputs]
CONFIDENCE: [0.0-1.0] because [brief justification]
CITATIONS: [list every section/page reference used]
""",

    # ── multi_doc — cross-period or cross-document analysis ───────────────────
    "multi_doc": """You are a precise financial analyst comparing data across periods or documents.
Your ONLY source of truth is the retrieved context below.
Never use training knowledge. Never guess.

RETRIEVED CONTEXT (your ONLY source — never use prior knowledge):
{{ retrieved_context }}

COMPANY: {{ company_name }}
FISCAL YEAR: {{ fiscal_year }}
QUERY TYPE: multi_doc (cross-period or cross-document comparison)

QUESTION: {{ question }}

STRICT RULES:
RULE 1: Answer ONLY from the retrieved context above.
RULE 2: If any comparison period is missing from context output RETRIEVAL_MISS.
RULE 3: Cite every number with its period: [SECTION / PAGE / PERIOD: value]
RULE 4: State units explicitly. Ensure consistent units across periods.
RULE 5: Clearly label each period in your comparison.
RULE 6: Note any restatements or methodology changes between periods.

Output format:
ANSWER: [comparison with figures for each period clearly labelled]
COMPUTATION: [delta calculation if applicable]
CONFIDENCE: [0.0-1.0] because [brief justification]
CITATIONS: [list every section/page/period reference used]
""",

    # ── text — narrative analysis ─────────────────────────────────────────────
    "text": """You are a precise financial analyst extracting and summarising narrative content.
Your ONLY source of truth is the retrieved context below.
Never use training knowledge. Never guess.

RETRIEVED CONTEXT (your ONLY source — never use prior knowledge):
{{ retrieved_context }}

COMPANY: {{ company_name }}
FISCAL YEAR: {{ fiscal_year }}
QUERY TYPE: text (narrative analysis)

QUESTION: {{ question }}

STRICT RULES:
RULE 1: Answer ONLY from the retrieved context above.
RULE 2: If the topic is not covered in context output RETRIEVAL_MISS.
RULE 3: Cite every claim: [SECTION_NAME / PAGE_NUM]
RULE 4: Do not extrapolate beyond what the document states.
RULE 5: Note any caveats or qualifications the document makes.
RULE 6: Never introduce external knowledge or opinions.

Output format:
ANSWER: [narrative answer with inline citations]
COMPUTATION: N/A
CONFIDENCE: [0.0-1.0] because [brief justification]
CITATIONS: [list every section/page reference used]
""",

    # ── forensic — anomaly and fraud signal detection ─────────────────────────
    "forensic": """You are a forensic financial analyst examining documents for anomalies.
Your ONLY source of truth is the retrieved context below.
Never use training knowledge. Never guess.

RETRIEVED CONTEXT (your ONLY source — never use prior knowledge):
{{ retrieved_context }}

COMPANY: {{ company_name }}
FISCAL YEAR: {{ fiscal_year }}
QUERY TYPE: forensic (anomaly detection)

QUESTION: {{ question }}

STRICT RULES:
RULE 1: Answer ONLY from the retrieved context above.
RULE 2: Flag signals as LOW / MEDIUM / HIGH risk — never conclude fraud directly.
RULE 3: Cite every figure used in your analysis: [SECTION_NAME / PAGE_NUM: value]
RULE 4: Compare figures explicitly: state what is expected vs what is observed.
RULE 5: Note any restatements, qualifications, or auditor concerns from context.
RULE 6: Forensic signals are hypotheses for investigation, not accusations.

Output format:
ANSWER: [forensic assessment with risk signals and evidence]
COMPUTATION: [quantitative analysis if applicable]
CONFIDENCE: [0.0-1.0] because [brief justification]
CITATIONS: [list every section/page reference used]
""",
}


# ── PromptAssembler ───────────────────────────────────────────────────────────

class PromptAssembler:
    """
    N10 Prompt Assembler.

    Builds C7-compliant prompts from retrieved chunks.
    Context ALWAYS before question — enforced at render time.

    Two usage modes:
        1. assembler.assemble(query, chunks, query_type, ...) → str
        2. assembler.run(ba_state)                            → BAState

    The assembler validates C7 compliance on every rendered prompt.
    If context does not appear before question, raises ValueError.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, object] = {}
        self._env = None

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N10 node entry point.

        Reads:
            state.query              — analyst question
            state.query_type         — from N04 CART Router
            state.retrieval_stage_2  — top-3 chunks from N09
            state.company_name       — from N01 ingestion
            state.fiscal_year        — from N01 ingestion

        Writes:
            state.assembled_prompt   — final C7-compliant prompt string
            state.prompt_template    — template name used (always 'context_first')

        Args:
            state: BAState object

        Returns:
            BAState with assembled_prompt populated
        """
        query      = getattr(state, "query",             "") or ""
        query_type = getattr(state, "query_type",        "text") or "text"
        chunks     = getattr(state, "retrieval_stage_2", []) or []
        company    = getattr(state, "company_name",      "UNKNOWN") or "UNKNOWN"
        fiscal_year= getattr(state, "fiscal_year",       "UNKNOWN") or "UNKNOWN"

        if not query:
            logger.warning("N10: empty query — skipping prompt assembly")
            state.assembled_prompt = ""
            state.prompt_template  = "context_first"
            return state

        prompt = self.assemble(
            query       = query,
            chunks      = chunks,
            query_type  = query_type,
            company_name= company,
            fiscal_year = fiscal_year,
        )

        state.assembled_prompt = prompt
        state.prompt_template  = "context_first"  # C7 — always context_first

        logger.info(
            "N10 Prompt: type=%s | chunks=%d | prompt_len=%d chars",
            query_type, len(chunks), len(prompt),
        )
        return state

    # ── Core assembly method ──────────────────────────────────────────────────

    def assemble(
        self,
        query:        str,
        chunks:       List[Dict],
        query_type:   str  = "text",
        company_name: str  = "UNKNOWN",
        fiscal_year:  str  = "UNKNOWN",
    ) -> str:
        """
        Build a C7-compliant prompt from retrieved chunks.

        Args:
            query        : Analyst question string
            chunks       : List of chunk dicts from retrieval_stage_2
            query_type   : One of numerical/ratio/multi_doc/text/forensic
            company_name : Company name from N01 ingestion
            fiscal_year  : Fiscal year from N01 ingestion

        Returns:
            Rendered prompt string with context before question (C7)

        Raises:
            ValueError: If rendered prompt violates C7
                        (question appears before context)
        """
        # Normalise query_type — fall back to text if unknown
        if query_type not in QUERY_TYPES:
            logger.warning(
                "N10: unknown query_type '%s' — falling back to 'text'",
                query_type,
            )
            query_type = "text"

        # Build retrieved context string from chunks
        retrieved_context = self._format_chunks(chunks)

        # Render template
        template = self._get_template(query_type)
        prompt   = template.render(
            retrieved_context = retrieved_context,
            question          = query,
            company_name      = company_name,
            fiscal_year       = fiscal_year,
        )

        # C7 enforcement — raise if violated
        self._assert_context_first(prompt, query)

        return prompt

    # ── Private helpers ───────────────────────────────────────────────────────

    def _format_chunks(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into a single context string.

        Each chunk is prefixed with its C8 metadata key and rank.
        Chunks are limited to MAX_CHUNKS and MAX_CHUNK_CHARS each.

        Args:
            chunks: List of chunk dicts from retrieval_stage_2

        Returns:
            Formatted context string
        """
        if not chunks:
            return "[No relevant context retrieved from document]"

        parts = []
        for i, chunk in enumerate(chunks[:MAX_CHUNKS]):
            text     = chunk.get("text", "") or chunk.get("page_content", "")
            company  = chunk.get("company",     "UNKNOWN")
            doc_type = chunk.get("doc_type",    "UNKNOWN")
            fy       = chunk.get("fiscal_year", "UNKNOWN")
            section  = chunk.get("section",     "UNKNOWN")
            page     = chunk.get("page",         0)
            chunk_id = chunk.get("chunk_id",    f"chunk_{i+1}")

            # C8 prefix: COMPANY/DOCTYPE/FISCAL_YEAR/SECTION/PAGE
            c8_prefix = f"{company}/{doc_type}/{fy}/{section}/{page}"

            # Truncate very long chunks
            if len(text) > MAX_CHUNK_CHARS:
                text = text[:MAX_CHUNK_CHARS] + "... [truncated]"

            part = (
                f"[CHUNK {i+1} | {c8_prefix}]\n"
                f"{text}\n"
            )
            parts.append(part)

        return "\n".join(parts)

    def _get_template(self, query_type: str):
        """Get compiled Jinja2 template for query type."""
        jinja2 = _get_jinja2()

        if not self._env:
            self._env = jinja2.Environment(
                trim_blocks   = True,
                lstrip_blocks = True,
                undefined     = jinja2.StrictUndefined,
            )

        if query_type not in self._templates:
            template_str             = TEMPLATES[query_type]
            self._templates[query_type] = self._env.from_string(template_str)

        return self._templates[query_type]

    @staticmethod
    def _assert_context_first(prompt: str, query: str) -> None:
        """
        C7 enforcement: assert retrieved_context appears before question.

        Finds position of RETRIEVED CONTEXT marker and QUESTION marker.
        Raises ValueError if question marker appears before context marker.

        Args:
            prompt : Rendered prompt string
            query  : Original question (for error message)

        Raises:
            ValueError: If C7 is violated
        """
        context_pos  = prompt.find(_CONTEXT_MARKER)
        question_pos = prompt.find(_QUESTION_MARKER)

        if context_pos == -1:
            raise ValueError(
                f"C7 VIOLATION: '{_CONTEXT_MARKER}' marker not found in prompt. "
                f"Every prompt must contain RETRIEVED CONTEXT before the question."
            )

        if question_pos == -1:
            raise ValueError(
                f"C7 VIOLATION: '{_QUESTION_MARKER}' marker not found in prompt. "
                f"Every prompt must contain a QUESTION section."
            )

        if question_pos < context_pos:
            raise ValueError(
                f"C7 VIOLATION: QUESTION (pos={question_pos}) appears before "
                f"RETRIEVED CONTEXT (pos={context_pos}). "
                f"Context must ALWAYS precede the question. "
                f"Query: '{query[:80]}'"
            )


# ── Convenience wrapper for LangGraph N10 node ───────────────────────────────

def run_prompt_assembler(state) -> object:
    """
    Convenience wrapper used by the LangGraph pipeline node N10.

    Args:
        state: BAState object

    Returns:
        BAState with assembled_prompt populated
    """
    assembler = PromptAssembler()
    return assembler.run(state)


def assemble_prompt(
    query:        str,
    chunks:       List[Dict],
    query_type:   str = "text",
    company_name: str = "UNKNOWN",
    fiscal_year:  str = "UNKNOWN",
) -> str:
    """
    Functional interface for prompt assembly.
    Used in tests and direct pipeline calls.
    """
    assembler = PromptAssembler()
    return assembler.assemble(
        query        = query,
        chunks       = chunks,
        query_type   = query_type,
        company_name = company_name,
        fiscal_year  = fiscal_year,
    )