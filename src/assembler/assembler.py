"""
src/assembler/assembler.py

Production-Grade Prompt Assembler

Optimized for:
- FinanceBench
- Long-context LLMs
- Hybrid retrieval
- Citation preservation
- Token safety
- Financial QA
- Windows
- Colab
- Ollama
"""

from __future__ import annotations

import logging
import re

from typing import Dict
from typing import List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONTEXT_CHUNKS = 10

MAX_CHUNK_CHARS = 3500

MAX_TOTAL_CONTEXT_CHARS = 24000

MIN_CONTEXT_CHARS = 200

# ─────────────────────────────────────────────────────────────────────────────
# Templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an elite financial analyst AI.

Your job:
- Answer ONLY using provided context.
- Use precise financial reasoning.
- Do NOT hallucinate.
- Prefer exact figures from filings.
- If answer is missing, say:
  INSUFFICIENT_EVIDENCE

Rules:
- Be concise.
- Preserve units.
- Preserve percentages.
- Preserve dates.
- Use direct financial terminology.
"""

USER_PROMPT_TEMPLATE = """
QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the question directly.
2. Use only evidence from context.
3. If multiple numbers exist, choose the most relevant one.
4. Preserve exact financial values.
5. Keep response under 120 words.

FINAL ANSWER:
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def normalize_whitespace(
    text: str,
) -> str:

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def clean_chunk_text(
    text: str,
) -> str:

    text = (
        text or ""
    ).strip()

    text = text.replace(
        "\x00",
        ""
    )

    text = normalize_whitespace(
        text
    )

    return text


def estimate_tokens(
    text: str,
) -> int:

    if not text:
        return 0

    return max(
        1,
        len(text) // 4,
    )


def build_chunk_header(
    chunk: Dict,
) -> str:

    company = chunk.get(
        "company",
        "UNKNOWN",
    )

    doc_type = chunk.get(
        "doc_type",
        "UNKNOWN",
    )

    fiscal_year = chunk.get(
        "fiscal_year",
        "UNKNOWN",
    )

    section = chunk.get(
        "section",
        "UNKNOWN",
    )

    page = chunk.get(
        "page",
        "?",
    )

    return (
        f"[{company} | "
        f"{doc_type} | "
        f"{fiscal_year} | "
        f"Section: {section} | "
        f"Page: {page}]"
    )


def deduplicate_chunks(
    chunks: List[Dict],
) -> List[Dict]:

    seen = set()

    output = []

    for chunk in chunks:

        text = clean_chunk_text(
            chunk.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        key = hash(text)

        if key in seen:
            continue

        seen.add(key)

        output.append(chunk)

    return output

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Assembler
# ─────────────────────────────────────────────────────────────────────────────


class PromptAssembler:

    def __init__(
        self,
        max_context_chunks: int = MAX_CONTEXT_CHUNKS,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        max_total_context_chars: int = MAX_TOTAL_CONTEXT_CHARS,
    ):

        self.max_context_chunks = max(
            1,
            int(max_context_chunks),
        )

        self.max_chunk_chars = max(
            500,
            int(max_chunk_chars),
        )

        self.max_total_context_chars = max(
            2000,
            int(max_total_context_chars),
        )

    # ─────────────────────────────────────────────────────────────────────

    def run(
        self,
        state,
    ):

        query = getattr(
            state,
            "query",
            "",
        ) or ""

        reranked_chunks = getattr(
            state,
            "reranked_chunks",
            [],
        ) or []

        sniper_hit = getattr(
            state,
            "sniper_hit",
            False,
        )

        sniper_answer = getattr(
            state,
            "sniper_answer",
            "",
        ) or ""

        if not query:

            logger.warning(
                "[ASSEMBLER] Empty query"
            )

            state.assembled_prompt = ""

            return state

        # Sniper-first

        if (
            sniper_hit
            and sniper_answer
        ):

            context = (
                "HIGH_CONFIDENCE_TABLE_MATCH\n\n"
                f"{sniper_answer}"
            )

            assembled = (
                USER_PROMPT_TEMPLATE.format(
                    question=query,
                    context=context,
                )
            )

            state.assembled_prompt = (
                assembled
            )

            state.prompt_tokens = (
                estimate_tokens(
                    assembled
                )
            )

            return state

        # Retrieval Context

        context = self.build_context(
            reranked_chunks
        )

        if (
            len(context)
            < MIN_CONTEXT_CHARS
        ):

            context += (
                "\n\nWARNING: "
                "Limited retrieval evidence."
            )

        assembled = (
            USER_PROMPT_TEMPLATE.format(
                question=query,
                context=context,
            )
        )

        final_prompt = (
            SYSTEM_PROMPT.strip()
            + "\n\n"
            + assembled.strip()
        )

        state.assembled_prompt = (
            final_prompt
        )

        state.prompt_tokens = (
            estimate_tokens(
                final_prompt
            )
        )

        logger.info(
            "[ASSEMBLER] chunks=%d tokens=%d",
            len(reranked_chunks),
            state.prompt_tokens,
        )

        return state

    # ─────────────────────────────────────────────────────────────────────

    def build_context(
        self,
        chunks: List[Dict],
    ) -> str:

        chunks = (
            deduplicate_chunks(
                chunks
            )
        )

        chunks = chunks[
            : self.max_context_chunks
        ]

        context_blocks = []

        current_chars = 0

        for idx, chunk in enumerate(
            chunks,
            start=1,
        ):

            text = clean_chunk_text(
                chunk.get(
                    "text",
                    "",
                )
            )

            if not text:
                continue

            text = text[
                : self.max_chunk_chars
            ]

            header = (
                build_chunk_header(
                    chunk
                )
            )

            score = (
                chunk.get(
                    "retrieval_confidence",
                    chunk.get(
                        "rrf_score",
                        chunk.get(
                            "bm25_score",
                            chunk.get(
                                "bge_score",
                                0.0,
                            ),
                        ),
                    ),
                )
            )

            block = (
                f"[CONTEXT {idx}]\n"
                f"{header}\n"
                f"Score: {score}\n\n"
                f"{text}"
            )

            projected = (
                current_chars
                + len(block)
            )

            if (
                projected
                > self.max_total_context_chars
            ):

                break

            context_blocks.append(
                block
            )

            current_chars += len(
                block
            )

        if not context_blocks:

            return (
                "NO_RELEVANT_CONTEXT_FOUND"
            )

        return "\n\n".join(
            context_blocks
        )

    # ─────────────────────────────────────────────────────────────────────

    def build_direct_prompt(
        self,
        question: str,
        context_chunks: List[Dict],
    ) -> str:

        context = self.build_context(
            context_chunks
        )

        prompt = (
            SYSTEM_PROMPT.strip()
            + "\n\n"
            + USER_PROMPT_TEMPLATE.format(
                question=question,
                context=context,
            )
        )

        return prompt.strip()

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