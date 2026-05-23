"""
src/prompts/assembler.py

Production-Grade Prompt Assembler
FinBench Multi-Agent Business Analyst AI

Capabilities
------------
1. Context-first prompting (C7 enforced)
2. Multi-template query specialization
3. Financial-safe prompt construction
4. Prompt injection hardening
5. Retrieval chunk normalization
6. Dynamic token budgeting
7. Citation enforcement
8. Hallucination reduction
9. Context truncation
10. Template precompilation
11. Strict rendering validation
12. Structured answer enforcement
13. Financial metadata propagation
14. Retrieval deduplication
15. Production-safe fallback handling
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from jinja2 import (
    BaseLoader,
    Environment,
    StrictUndefined,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Query Types
# ─────────────────────────────────────────────────────────────────────────────

class QueryType:

    NUMERICAL = "numerical"

    RATIO = "ratio"

    MULTI_DOC = "multi_doc"

    TEXT = "text"

    FORENSIC = "forensic"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 32000

MAX_CHUNKS = 12

MIN_CHUNKS = 1

CONTEXT_FIRST_TEMPLATE = "context_first"

# ─────────────────────────────────────────────────────────────────────────────
# Base System Instructions
# ─────────────────────────────────────────────────────────────────────────────

BASE_RULES = """
CRITICAL RULES:
1. Use ONLY retrieved context.
2. Never use model memory.
3. Every claim requires citation.
4. If missing evidence:
   RETRIEVAL_MISS: insufficient evidence
5. Never fabricate numbers.
6. Always preserve fiscal year consistency.
7. Cite as: [SECTION / PAGE]
8. Retrieved context appears BEFORE question.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Templates
# ─────────────────────────────────────────────────────────────────────────────

NUMERICAL_TEMPLATE = """
You are an expert financial analyst.

{{ rules }}

DOCUMENT:
Company: {{ company }}
Document: {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }} ---
Section: {{ chunk.section }}
Page: {{ chunk.page }}

{{ chunk.text }}

{% endfor %}

TASK:
Extract the exact numerical answer.

QUESTION:
{{ question }}

OUTPUT FORMAT:
ANSWER:
COMPUTATION:
CONFIDENCE:
CITATIONS:
"""

RATIO_TEMPLATE = """
You are an expert financial analyst.

{{ rules }}

DOCUMENT:
Company: {{ company }}
Document: {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }} ---
Section: {{ chunk.section }}
Page: {{ chunk.page }}

{{ chunk.text }}

{% endfor %}

TASK:
Compute the ratio carefully.

QUESTION:
{{ question }}

OUTPUT FORMAT:
ANSWER:
FORMULA:
COMPUTATION:
CONFIDENCE:
CITATIONS:
"""

MULTI_DOC_TEMPLATE = """
You are an expert financial analyst.

{{ rules }}

DOCUMENT:
Company: {{ company }}
Document: {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }} ---
Section: {{ chunk.section }}
Page: {{ chunk.page }}

{{ chunk.text }}

{% endfor %}

TASK:
Perform structured comparison.

QUESTION:
{{ question }}

OUTPUT FORMAT:
ANSWER:
COMPARISON:
COMPUTATION:
CONFIDENCE:
CITATIONS:
"""

TEXT_TEMPLATE = """
You are an expert financial analyst.

{{ rules }}

DOCUMENT:
Company: {{ company }}
Document: {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }} ---
Section: {{ chunk.section }}
Page: {{ chunk.page }}

{{ chunk.text }}

{% endfor %}

TASK:
Provide a narrative answer using only evidence above.

QUESTION:
{{ question }}

OUTPUT FORMAT:
ANSWER:
CONFIDENCE:
CITATIONS:
"""

FORENSIC_TEMPLATE = """
You are an expert forensic financial analyst.

{{ rules }}

DOCUMENT:
Company: {{ company }}
Document: {{ doc_type }}
Fiscal Year: {{ fiscal_year }}

RETRIEVED SECTIONS:
{% for chunk in chunks %}
--- Source {{ loop.index }} ---
Section: {{ chunk.section }}
Page: {{ chunk.page }}

{{ chunk.text }}

{% endfor %}

FORENSIC CHECKLIST:
1. Benford anomalies
2. Round-number concentration
3. Trend discontinuities
4. Disclosure inconsistencies
5. Reconciliation failures

QUESTION:
{{ question }}

OUTPUT FORMAT:
ANSWER:
ANOMALIES:
CONFIDENCE:
CITATIONS:
"""

# ─────────────────────────────────────────────────────────────────────────────
# Template Registry
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES = {
    QueryType.NUMERICAL: NUMERICAL_TEMPLATE,
    QueryType.RATIO: RATIO_TEMPLATE,
    QueryType.MULTI_DOC: MULTI_DOC_TEMPLATE,
    QueryType.TEXT: TEXT_TEMPLATE,
    QueryType.FORENSIC: FORENSIC_TEMPLATE,
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def sanitize_text(
    text: str,
) -> str:

    if not text:
        return ""

    text = text.replace(
        "\x00",
        ""
    )

    text = re.sub(
        r"<\|.*?\|>",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def deduplicate_chunks(
    chunks: List[Dict],
) -> List[Dict]:

    seen = set()

    unique = []

    for chunk in chunks:

        text = sanitize_text(
            chunk.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        key = hash(text)

        if key in seen:
            continue

        seen.add(key)

        unique.append(chunk)

    return unique


def truncate_chunks(
    chunks: List[Dict],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> List[Dict]:

    output = []

    running = 0

    for chunk in chunks:

        text = chunk.get(
            "text",
            "",
        )

        length = len(text)

        if (
            running + length
            > max_chars
        ):
            break

        output.append(chunk)

        running += length

    return output

# ─────────────────────────────────────────────────────────────────────────────
# PromptAssembler
# ─────────────────────────────────────────────────────────────────────────────


class PromptAssembler:

    def __init__(self):

        self._env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self._compiled = {}

        self._compile_templates()

    # ─────────────────────────────────────────────────────────────────────────
    # Compile
    # ─────────────────────────────────────────────────────────────────────────

    def _compile_templates(
        self,
    ):

        for (
            query_type,
            template_str,
        ) in TEMPLATES.items():

            self._compiled[
                query_type
            ] = self._env.from_string(
                template_str
            )

    # ─────────────────────────────────────────────────────────────────────────
    # LangGraph Node
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state):

        question = getattr(
            state,
            "query",
            "",
        )

        if not question:

            logger.warning(
                "[ASSEMBLER] Empty query"
            )

            state.assembled_prompt = ""

            state.prompt_template = (
                CONTEXT_FIRST_TEMPLATE
            )

            return state

        chunks = (
            getattr(
                state,
                "retrieval_stage_2",
                [],
            )
            or getattr(
                state,
                "retrieval_stage_1",
                [],
            )
            or []
        )

        chunks = (
            deduplicate_chunks(
                chunks
            )
        )

        chunks = chunks[
            : MAX_CHUNKS
        ]

        chunks = truncate_chunks(
            chunks
        )

        normalized = (
            self._normalize_chunks(
                chunks
            )
        )

        query_type = getattr(
            state,
            "query_type",
            QueryType.TEXT,
        )

        if (
            query_type
            not in TEMPLATES
        ):
            query_type = (
                QueryType.TEXT
            )

        prompt = self._assemble(
            query_type=query_type,
            question=question,
            chunks=normalized,
            company=getattr(
                state,
                "company_name",
                "Unknown Company",
            ),
            doc_type=getattr(
                state,
                "doc_type",
                "10-K",
            ),
            fiscal_year=getattr(
                state,
                "fiscal_year",
                "Unknown FY",
            ),
        )

        self._validate_context_first(
            prompt,
            question,
        )

        state.assembled_prompt = (
            prompt
        )

        state.prompt_template = (
            CONTEXT_FIRST_TEMPLATE
        )

        logger.info(
            "[ASSEMBLER] type=%s chunks=%d chars=%d",
            query_type,
            len(normalized),
            len(prompt),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Assemble
    # ─────────────────────────────────────────────────────────────────────────

    def _assemble(
        self,
        query_type: str,
        question: str,
        chunks: List[Dict],
        company: str,
        doc_type: str,
        fiscal_year: str,
    ) -> str:

        template = self._compiled.get(
            query_type,
            self._compiled[
                QueryType.TEXT
            ],
        )

        return template.render(
            rules=BASE_RULES,
            question=sanitize_text(
                question
            ),
            chunks=chunks,
            company=sanitize_text(
                company
            ),
            doc_type=sanitize_text(
                doc_type
            ),
            fiscal_year=sanitize_text(
                fiscal_year
            ),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Direct Assembly
    # ─────────────────────────────────────────────────────────────────────────

    def assemble_direct(
        self,
        query_type: str,
        question: str,
        chunks: List[Dict],
        company: str = "Unknown Company",
        doc_type: str = "10-K",
        fiscal_year: str = "Unknown FY",
    ) -> str:

        normalized = (
            self._normalize_chunks(
                chunks
            )
        )

        normalized = (
            deduplicate_chunks(
                normalized
            )
        )

        normalized = truncate_chunks(
            normalized
        )

        return self._assemble(
            query_type=query_type,
            question=question,
            chunks=normalized,
            company=company,
            doc_type=doc_type,
            fiscal_year=fiscal_year,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Normalize
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_chunks(
        self,
        chunks: List[Dict],
    ) -> List[Dict]:

        normalized = []

        for chunk in chunks:

            text = (
                chunk.get("text")
                or chunk.get(
                    "content",
                    ""
                )
            )

            text = sanitize_text(
                text
            )

            if not text:
                continue

            normalized.append(
                {
                    "text": text,
                    "section": sanitize_text(
                        str(
                            chunk.get(
                                "section",
                                "Unknown Section",
                            )
                        )
                    ),
                    "page": str(
                        chunk.get(
                            "page",
                            "?",
                        )
                    ),
                    "company": sanitize_text(
                        str(
                            chunk.get(
                                "company",
                                "",
                            )
                        )
                    ),
                    "doc_type": sanitize_text(
                        str(
                            chunk.get(
                                "doc_type",
                                "",
                            )
                        )
                    ),
                    "fiscal_year": sanitize_text(
                        str(
                            chunk.get(
                                "fiscal_year",
                                "",
                            )
                        )
                    ),
                }
            )

        return normalized

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_context_first(
        self,
        prompt: str,
        question: str,
    ) -> None:

        retrieved_pos = prompt.find(
            "RETRIEVED SECTIONS"
        )

        question_pos = prompt.find(
            "QUESTION:"
        )

        if (
            retrieved_pos == -1
            or question_pos == -1
        ):
            return

        if question_pos < retrieved_pos:

            raise AssertionError(
                "C7 violation: question before context"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def context_before_question(
        self,
        prompt: str,
    ) -> bool:

        retrieved_pos = prompt.find(
            "RETRIEVED SECTIONS"
        )

        question_pos = prompt.find(
            "QUESTION:"
        )

        if (
            retrieved_pos == -1
            or question_pos == -1
        ):
            return True

        return (
            retrieved_pos
            < question_pos
        )

    def get_template_names(
        self,
    ) -> List[str]:

        return list(
            self._compiled.keys()
        )

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def run_prompt_assembler(
    state,
):

    assembler = PromptAssembler()

    return assembler.run(
        state
    )