"""
src/ingestion/section_tree_builder.py

Production-Grade Section Tree Builder
FinBench Multi-Agent Business Analyst AI

Capabilities
------------
1. Hierarchical SEC section parsing
2. Font-aware heading detection
3. Multi-level tree construction
4. Iterative stack-based nesting
5. SEC filing section classification
6. Memory-safe summarization
7. Local LLM summaries
8. Noise filtering
9. Duplicate heading removal
10. Robust page-range assignment
11. Dynamic hierarchy inference
12. Production-safe fallback logic
13. Financial document specialization
14. Retrieval-friendly metadata
15. Large-document optimization
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SEC Section Types
# ─────────────────────────────────────────────────────────────────────────────

SEC_MAJOR_SECTIONS = {
    "business": "Business Overview",
    "risk factor": "Risk Factors",
    "management": "MD&A",
    "financial statement": "Financial Statements",
    "note": "Notes",
    "quantitative": "Quantitative Disclosures",
    "controls": "Controls and Procedures",
    "legal": "Legal Proceedings",
    "market": "Market Information",
    "executive compensation": "Executive Compensation",
    "security ownership": "Security Ownership",
    "properties": "Properties",
}

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

HEADING_H1_MIN = 16.0

HEADING_H2_MIN = 13.0

HEADING_H3_MIN = 11.5

MAX_HEADING_LENGTH = 200

MIN_HEADING_LENGTH = 3

MAX_SECTIONS_TO_SUMMARISE = 8

MAX_SUMMARY_SNIPPET = 1000

MAX_LAST_SECTION_SPAN = 50

# ─────────────────────────────────────────────────────────────────────────────
# SectionNode
# ─────────────────────────────────────────────────────────────────────────────


class SectionNode:

    __slots__ = (
        "name",
        "level",
        "start_page",
        "end_page",
        "font_size",
        "is_bold",
        "summary",
        "sec_type",
        "children",
    )

    def __init__(
        self,
        name: str,
        level: int,
        start_page: int,
        end_page: int = 0,
        font_size: float = 13.0,
        is_bold: bool = False,
        summary: str = "",
        sec_type: str = "",
    ):

        self.name = name

        self.level = level

        self.start_page = start_page

        self.end_page = end_page

        self.font_size = font_size

        self.is_bold = is_bold

        self.summary = summary

        self.sec_type = sec_type

        self.children: List[
            SectionNode
        ] = []

    def to_dict(self):

        return {
            "name": self.name,
            "level": self.level,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "font_size": self.font_size,
            "is_bold": self.is_bold,
            "summary": self.summary,
            "sec_type": self.sec_type,
            "children": [
                c.to_dict()
                for c in self.children
            ],
        }

# ─────────────────────────────────────────────────────────────────────────────
# SectionTreeBuilder
# ─────────────────────────────────────────────────────────────────────────────


class SectionTreeBuilder:

    def __init__(
        self,
        llm_client=None,
    ):

        self._llm = llm_client

    # ─────────────────────────────────────────────────────────────────────────
    # LangGraph Node
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, state):

        headings = getattr(
            state,
            "heading_positions",
            [],
        ) or []

        raw_text = getattr(
            state,
            "raw_text",
            "",
        ) or ""

        section_tree = self.build(
            headings,
            raw_text,
        )

        state.section_tree = (
            section_tree
        )

        logger.info(
            "[SECTION_TREE] sections=%d",
            section_tree.get(
                "total_sections",
                0,
            ),
        )

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────────

    def build(
        self,
        heading_positions: List[Dict],
        raw_text: str = "",
    ) -> Dict:

        headings = (
            self._clean_headings(
                heading_positions
            )
        )

        if not headings:
            return self._empty_tree()

        headings = (
            self._assign_levels(
                headings
            )
        )

        headings = (
            self._assign_page_ranges(
                headings
            )
        )

        nodes = self._build_tree(
            headings
        )

        self._classify_sections(
            nodes
        )

        if (
            self._llm
            and raw_text
        ):

            self._add_summaries(
                nodes,
                raw_text,
            )

        return {
            "document": "root",
            "total_sections": self._count_sections(
                nodes
            ),
            "children": [
                n.to_dict()
                for n in nodes
            ],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Clean
    # ─────────────────────────────────────────────────────────────────────────

    def _clean_headings(
        self,
        headings: List[Dict],
    ) -> List[Dict]:

        cleaned = []

        seen = set()

        for h in headings:

            text = (
                h.get(
                    "text",
                    "",
                )
                .strip()
            )

            if not text:
                continue

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            if (
                len(text)
                < MIN_HEADING_LENGTH
            ):
                continue

            if (
                len(text)
                > MAX_HEADING_LENGTH
            ):
                continue

            if re.match(
                r"^[\d\W]+$",
                text,
            ):
                continue

            key = (
                text.lower(),
                int(
                    h.get(
                        "page",
                        0,
                    )
                ),
            )

            if key in seen:
                continue

            seen.add(key)

            cleaned.append(
                {
                    **h,
                    "text": text,
                }
            )

        cleaned.sort(
            key=lambda x: (
                x.get(
                    "page",
                    0,
                ),
                -float(
                    x.get(
                        "font_size",
                        0,
                    )
                ),
            )
        )

        return cleaned

    # ─────────────────────────────────────────────────────────────────────────
    # Levels
    # ─────────────────────────────────────────────────────────────────────────

    def _assign_levels(
        self,
        headings: List[Dict],
    ) -> List[Dict]:

        output = []

        for h in headings:

            font_size = float(
                h.get(
                    "font_size",
                    13.0,
                )
            )

            is_bold = bool(
                h.get(
                    "is_bold",
                    False,
                )
            )

            if (
                font_size
                >= HEADING_H1_MIN
            ):

                level = 1

            elif (
                font_size
                >= HEADING_H2_MIN
            ):

                level = (
                    1
                    if is_bold
                    else 2
                )

            elif (
                font_size
                >= HEADING_H3_MIN
            ):

                level = 3

            else:

                level = 4

            output.append(
                {
                    **h,
                    "level": level,
                }
            )

        return output

    # ─────────────────────────────────────────────────────────────────────────
    # Page Ranges
    # ─────────────────────────────────────────────────────────────────────────

    def _assign_page_ranges(
        self,
        headings: List[Dict],
    ) -> List[Dict]:

        output = []

        for idx, h in enumerate(
            headings
        ):

            start_page = int(
                h.get(
                    "page",
                    0,
                )
            )

            if idx + 1 < len(
                headings
            ):

                next_page = int(
                    headings[
                        idx + 1
                    ].get(
                        "page",
                        start_page,
                    )
                )

                end_page = max(
                    start_page,
                    next_page - 1,
                )

            else:

                end_page = (
                    start_page
                    + MAX_LAST_SECTION_SPAN
                )

            output.append(
                {
                    **h,
                    "start_page": start_page,
                    "end_page": end_page,
                }
            )

        return output

    # ─────────────────────────────────────────────────────────────────────────
    # Tree
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tree(
        self,
        headings: List[Dict],
    ) -> List[SectionNode]:

        roots: List[
            SectionNode
        ] = []

        stack: List[
            SectionNode
        ] = []

        for heading in headings:

            node = SectionNode(
                name=heading.get(
                    "text",
                    "",
                ),
                level=heading.get(
                    "level",
                    1,
                ),
                start_page=heading.get(
                    "start_page",
                    0,
                ),
                end_page=heading.get(
                    "end_page",
                    0,
                ),
                font_size=float(
                    heading.get(
                        "font_size",
                        13.0,
                    )
                ),
                is_bold=bool(
                    heading.get(
                        "is_bold",
                        False,
                    )
                ),
            )

            while (
                stack
                and stack[-1].level
                >= node.level
            ):
                stack.pop()

            if stack:

                stack[-1].children.append(
                    node
                )

            else:

                roots.append(node)

            stack.append(node)

        return roots

    # ─────────────────────────────────────────────────────────────────────────
    # Summaries
    # ─────────────────────────────────────────────────────────────────────────

    def _add_summaries(
        self,
        nodes: List[SectionNode],
        raw_text: str,
    ) -> None:

        count = 0

        for node in self._walk(
            nodes
        ):

            if (
                count
                >= MAX_SECTIONS_TO_SUMMARISE
            ):
                break

            try:

                summary = (
                    self._generate_summary(
                        node.name,
                        raw_text,
                    )
                )

                node.summary = summary

                count += 1

            except Exception as exc:

                logger.debug(
                    "[SECTION_TREE] Summary failed: %s",
                    exc,
                )

    def _generate_summary(
        self,
        section_name: str,
        raw_text: str,
    ) -> str:

        if not self._llm:
            return ""

        idx = raw_text.lower().find(
            section_name.lower()[:50]
        )

        if idx == -1:

            snippet = raw_text[
                :MAX_SUMMARY_SNIPPET
            ]

        else:

            snippet = raw_text[
                idx:
                idx
                + MAX_SUMMARY_SNIPPET
            ]

        prompt = f"""
Summarize this SEC filing section in exactly one sentence.

SECTION:
{section_name}

TEXT:
{snippet}

RULES:
- One sentence only
- Financial factual tone
- No speculation
"""

        try:

            response = self._llm.chat(
                prompt,
                temperature=0.1,
            )

        except TypeError:

            response = self._llm.chat(
                prompt
            )

        response = (
            response.strip()
        )

        response = re.split(
            r"[.!?]",
            response,
        )[0].strip()

        if not response:
            return ""

        return (
            response[:300]
            + "."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SEC Classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_sections(
        self,
        nodes: List[SectionNode],
    ) -> None:

        for node in self._walk(
            nodes
        ):

            node.sec_type = (
                self._get_sec_type(
                    node.name
                )
            )

    @staticmethod
    def _get_sec_type(
        name: str,
    ) -> str:

        text = name.lower()

        for (
            key,
            sec_type,
        ) in (
            SEC_MAJOR_SECTIONS.items()
        ):

            if key in text:
                return sec_type

        return "Other"

    # ─────────────────────────────────────────────────────────────────────────
    # Walk
    # ─────────────────────────────────────────────────────────────────────────

    def _walk(
        self,
        nodes: List[SectionNode],
    ):

        stack = list(nodes)

        while stack:

            node = stack.pop(0)

            yield node

            stack = (
                node.children
                + stack
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Counts
    # ─────────────────────────────────────────────────────────────────────────

    def _count_sections(
        self,
        nodes: List[SectionNode],
    ) -> int:

        total = 0

        for _ in self._walk(
            nodes
        ):
            total += 1

        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Empty
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_tree():

        return {
            "document": "root",
            "total_sections": 0,
            "children": [],
        }

# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────


def run_section_tree_builder(
    state,
    llm_client=None,
):

    builder = SectionTreeBuilder(
        llm_client=llm_client
    )

    return builder.run(state)