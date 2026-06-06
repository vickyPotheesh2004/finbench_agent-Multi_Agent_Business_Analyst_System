"""
src/analysis/narrative_extractor.py
N20b — Narrative Extractor (Causal "What drove X?" Questions, OFFLINE)

Handles questions like:
  - "What drove operating margin change as of FY2022?"
  - "Why did revenue decline?"
  - "What caused the increase in interest expense?"
  - "Which segment had the lowest growth?"

Strategy:
  - NO LLM. Pure clause-level text mining.
  - Identify the SUBJECT metric (e.g. operating margin).
  - Locate the SUBJECT in MD&A-style text via raw_text + chunks.
  - Find sentences containing causal markers ("due to", "primarily",
    "driven by", "as a result of", "because of").
  - Extract DRIVER PHRASES (impairment, litigation, divestiture, FX,
    pricing, mix, volume, restructuring, …) from those sentences.
  - Return a clean comma-separated list of drivers + the source sentence.

Audit trail: every driver carries its source sentence and char-offset.

100% deterministic. ZERO LLM. Compatible with logic_lib.narrative_reasoner
rule 'extract_driver_list' (we are its implementation).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Narrative trigger vocabulary  (curated from FinanceBench gold answers
# and common 10-K MD&A language)
# ─────────────────────────────────────────────────────────────────────────────

# Causal markers — sentences with these are usually explaining drivers
CAUSAL_MARKERS = (
    "due to",
    "primarily due to",
    "primarily driven by",
    "driven by",
    "attributable to",
    "as a result of",
    "because of",
    "principally due to",
    "primarily reflecting",
    "reflecting",
    "resulting from",
    "impacted by",
    "affected by",
    "offset by",
    "partially offset by",
    "led by",
    "primarily",
    "largely due to",
    "mainly due to",
    "thanks to",
)

# Driver keyword categories — each driver phrase is normalised to a label
DRIVER_VOCABULARY: Dict[str, List[str]] = {
    # ── M&A / Portfolio ─────────────────────────────────────────────────
    "divestitures": [
        "divestiture", "divestitures", "divested", "divesting",
        "sale of", "disposition", "disposal of", "spin-off", "spinoff",
    ],
    "acquisitions": [
        "acquisition", "acquisitions", "acquired", "purchase of",
        "m&a", "mergers and acquisitions",
    ],
    # ── One-time / Restructuring ────────────────────────────────────────
    "litigation": [
        "litigation", "lawsuit", "legal settlement", "legal proceedings",
        "settlement charges", "settlement of", "combat arms", "earplug",
    ],
    "restructuring": [
        "restructuring", "restructure", "workforce reduction", "layoffs",
        "severance", "workforce action",
    ],
    "impairment": [
        "impairment", "impaired", "write-down", "writedown", "write-off",
        "writeoff", "goodwill impairment",
    ],
    "pfas_exit": [
        "pfas", "exit from pfas", "exit of pfas", "pfas exit",
        "phasing out pfas",
    ],
    "exit_market": [
        "exit from russia", "russia exit", "exit of business",
        "exit from", "wind down", "wind-down", "discontinued operation",
    ],
    # ── Macro / FX ──────────────────────────────────────────────────────
    "foreign_exchange": [
        "foreign exchange", "fx", "currency", "forex", "translation",
        "stronger dollar", "weaker dollar", "currency headwind",
        "currency tailwind", "fx headwind", "fx tailwind",
    ],
    "inflation": [
        "inflation", "inflationary", "rising costs", "cost increases",
        "raw material cost", "input cost inflation",
    ],
    "supply_chain": [
        "supply chain", "supply disruption", "shortages",
        "logistics costs", "freight costs",
    ],
    # ── Pricing / Volume / Mix ──────────────────────────────────────────
    "pricing": [
        "pricing actions", "price increase", "price decrease",
        "pricing", "selling price", "price realization",
    ],
    "volume": [
        "volume", "lower volume", "higher volume", "volume decline",
        "volume growth", "unit volume",
    ],
    "mix": [
        "product mix", "channel mix", "geographic mix", "mix shift",
        "favorable mix", "unfavorable mix", "mix",
    ],
    "demand": [
        "lower demand", "higher demand", "weaker demand", "stronger demand",
        "demand softness", "demand decline", "consumer demand",
    ],
    # ── Cost / Margin ───────────────────────────────────────────────────
    "cost_savings": [
        "cost savings", "productivity gains", "operational efficiency",
        "cost reduction", "savings initiatives",
    ],
    "operating_leverage": [
        "operating leverage", "deleveraging", "leverage on fixed costs",
    ],
    "investment": [
        "increased investment", "investment in r&d", "increased r&d",
        "marketing investment", "growth investment",
    ],
    # ── Tax ─────────────────────────────────────────────────────────────
    "tax_effects": [
        "tax rate", "discrete tax", "tax benefit", "tax expense",
        "deferred tax", "tax law change",
    ],
    # ── Pandemic / external events ──────────────────────────────────────
    "covid": [
        "covid", "covid-19", "pandemic", "post-pandemic", "covid related",
    ],
    "war_geopolitical": [
        "ukraine", "russia", "war", "sanctions", "geopolitical",
    ],
}

# Subject patterns for "what drove X" questions
SUBJECT_PATTERNS = [
    (r"operating\s+margin", "operating margin"),
    (r"gross\s+margin", "gross margin"),
    (r"net\s+margin", "net margin"),
    (r"net\s+income", "net income"),
    (r"operating\s+income", "operating income"),
    (r"revenue\s+(growth|change|decline|decrease|increase)?", "revenue"),
    (r"sales\s+(growth|change|decline|decrease|increase)?", "sales"),
    (r"earnings", "earnings"),
    (r"capex|capital\s+expenditure", "capex"),
    (r"cash\s+flow", "cash flow"),
    (r"interest\s+expense", "interest expense"),
    (r"r&d|research\s+and\s+development", "R&D"),
    (r"sg&a|selling.*administrative", "SG&A"),
]

# Causal question patterns — "what drove X", "why did Y", "what caused Z"
CAUSAL_QUESTION_PATTERNS = [
    r"\bwhat\s+drove\s+(.*?)(?:\s+(?:change|in|for|as|during|of|\?))",
    r"\bwhat\s+drove\b",
    r"\bwhy\s+did\s+",
    r"\bwhat\s+caused\s+",
    r"\bwhat\s+led\s+to\s+",
    r"\bwhat\s+were?\s+the\s+(main\s+)?(drivers|reasons|factors)\s+",
    r"\bwhat\s+factors\s+(contributed|drove|caused|affected)",
    r"\breason(s)?\s+for\b",
    r"\bdrivers\s+of\b",
    r"\bexplain\b",
]


# Segment patterns
SEGMENT_QUESTION_PATTERNS = [
    r"\bwhich\s+segment\s+",
    r"\bwhich\s+division\s+",
    r"\bsegment\s+(had|with|showed|drove)",
    r"\b(highest|lowest|best|worst)\s+(segment|division|growing\s+segment)",
]


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitter (no nltk dependency)
# ─────────────────────────────────────────────────────────────────────────────

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Basic split; good enough for MD&A prose
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_causal_question(question: str) -> bool:
    if not question:
        return False
    q = question.lower()
    return any(re.search(p, q) for p in CAUSAL_QUESTION_PATTERNS)


def detect_segment_question(question: str) -> bool:
    if not question:
        return False
    q = question.lower()
    return any(re.search(p, q) for p in SEGMENT_QUESTION_PATTERNS)


def detect_subject(question: str) -> Optional[str]:
    """Return the subject metric (e.g. 'operating margin')."""
    if not question:
        return None
    q = question.lower()
    for pat, label in SUBJECT_PATTERNS:
        if re.search(pat, q):
            return label
    return None


def find_drivers_in_sentence(sentence: str) -> List[Tuple[str, str]]:
    """
    Return list of (driver_category, matched_phrase) for drivers found in
    the given sentence. Empty list if none found.
    """
    if not sentence:
        return []
    s = sentence.lower()
    found = []
    for category, keywords in DRIVER_VOCABULARY.items():
        for kw in keywords:
            if kw in s:
                found.append((category, kw))
                break  # one match per category per sentence
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NarrativeAnswer:
    answered: bool
    output: str = ""
    drivers: List[str] = field(default_factory=list)
    evidence_sentences: List[str] = field(default_factory=list)
    confidence: float = 0.0
    audit_trail: Dict[str, Any] = field(default_factory=dict)


def _gather_search_text(
    raw_text: str,
    bm25_results: List[Dict],
    subject: str,
    max_chars: int = 80000,
) -> str:
    """
    Build the corpus to search in.

    FIX-v5 (2026-06-05): ALWAYS include raw_text windows around the
    subject (previously only as fallback when BM25 corpus was small).
    This is what made narrative_extractor's self-test pass on synthetic
    text but FAIL on the real 3M PDF — BM25 retrieved tables instead
    of MD&A sentences, and we never fell back to raw_text.
    """
    parts: List[str] = []

    # Source 1: BM25 chunks (ranked for query, usually best signal)
    if bm25_results:
        for c in bm25_results[:10]:
            t = c.get("text") or c.get("page_content") or ""
            if t:
                parts.append(t)

    # Source 2: ALWAYS scan raw_text around subject mentions (was conditional)
    if raw_text:
        for m in re.finditer(re.escape(subject.lower()), raw_text.lower()):
            start = max(0, m.start() - 500)
            end   = min(len(raw_text), m.end() + 2000)
            parts.append(raw_text[start:end])

    corpus = "\n\n".join(parts)
    if len(corpus) > 100:
        logger.info(
            "[narrative_extractor] corpus built: %d chars from %d BM25 chunks + raw_text",
            len(corpus), len(bm25_results) if bm25_results else 0,
        )
    return corpus[:max_chars]


def extract_drivers(
    question: str,
    raw_text: str,
    bm25_results: List[Dict],
    company: str = "",
    fiscal_year: str = "",
    doc_type: str = "",
) -> NarrativeAnswer:
    """
    Main entry: "what drove X" / "why Y" causal questions.

    Returns NarrativeAnswer with a comma-separated list of drivers
    extracted from sentences containing causal markers + subject.
    """
    if not detect_causal_question(question):
        return NarrativeAnswer(answered=False)

    subject = detect_subject(question)
    if not subject:
        return NarrativeAnswer(
            answered=False,
            audit_trail={"reason": "could not detect subject metric"},
        )

    logger.info("[narrative_extractor] Causal question, subject='%s'", subject)

    corpus = _gather_search_text(raw_text, bm25_results, subject)
    if not corpus:
        return NarrativeAnswer(
            answered=False,
            audit_trail={"reason": "empty corpus"},
        )

    # Sentences mentioning the subject
    sentences = _split_sentences(corpus)
    subject_lc = subject.lower()
    subj_sentences = [s for s in sentences if subject_lc in s.lower()]

    # Add sentences that immediately follow a subject sentence (drivers often
    # appear in the next sentence)
    extended: List[str] = []
    sent_set = {id(s) for s in subj_sentences}
    for i, s in enumerate(sentences):
        if id(s) in sent_set:
            extended.append(s)
            if i + 1 < len(sentences):
                extended.append(sentences[i + 1])
    if not extended:
        extended = subj_sentences

    # Filter to sentences with at least ONE causal marker
    causal_sentences = []
    for s in extended:
        s_lc = s.lower()
        if any(marker in s_lc for marker in CAUSAL_MARKERS):
            causal_sentences.append(s)

    if not causal_sentences:
        # Fallback: ANY sentence mentioning the subject
        causal_sentences = subj_sentences[:5]

    if not causal_sentences:
        return NarrativeAnswer(
            answered=False,
            audit_trail={"reason": "no causal/subject sentences found"},
        )

    # ── Mine drivers ─────────────────────────────────────────────────────
    driver_hits: Dict[str, List[str]] = {}     # category → list of evidence sentences
    for s in causal_sentences[:20]:            # cap scan to 20 sentences
        for category, _ in find_drivers_in_sentence(s):
            driver_hits.setdefault(category, []).append(s)

    if not driver_hits:
        # Fallback: return the most informative causal sentence verbatim
        best_sentence = max(causal_sentences, key=len)
        return NarrativeAnswer(
            answered=True,
            output=best_sentence[:400],
            drivers=[],
            evidence_sentences=[best_sentence],
            confidence=0.4,
            audit_trail={
                "subject": subject,
                "reason": "no driver vocabulary match; returning best causal sentence",
            },
        )

    # Sort drivers by frequency (most-mentioned first)
    sorted_drivers = sorted(
        driver_hits.keys(),
        key=lambda c: -len(driver_hits[c]),
    )

    # Human-readable driver labels
    driver_label_map = {
        "divestitures":       "divestitures",
        "acquisitions":       "acquisitions",
        "litigation":         "legal/litigation charges",
        "restructuring":      "restructuring",
        "impairment":         "impairment charges",
        "pfas_exit":          "PFAS exit",
        "exit_market":        "market exits / wind-downs",
        "foreign_exchange":   "foreign exchange",
        "inflation":          "inflation / rising input costs",
        "supply_chain":       "supply chain costs",
        "pricing":            "pricing actions",
        "volume":             "volume changes",
        "mix":                "product/segment mix",
        "demand":             "demand changes",
        "cost_savings":       "cost savings",
        "operating_leverage": "operating leverage",
        "investment":         "increased investment",
        "tax_effects":        "tax effects",
        "covid":              "COVID-19 impacts",
        "war_geopolitical":   "Russia/Ukraine and geopolitical impacts",
    }

    drivers_human = [driver_label_map.get(c, c) for c in sorted_drivers]

    # Build the final answer string
    if len(drivers_human) == 1:
        output = f"{subject.capitalize()} was driven by {drivers_human[0]}."
    elif len(drivers_human) == 2:
        output = f"{subject.capitalize()} was driven by {drivers_human[0]} and {drivers_human[1]}."
    else:
        head = ", ".join(drivers_human[:-1])
        output = f"{subject.capitalize()} was driven by {head}, and {drivers_human[-1]}."

    # Append evidence (one short citation)
    evidence_top = list({s: None for s in
                        [driver_hits[c][0] for c in sorted_drivers]
                        }.keys())[:3]

    if evidence_top:
        first_sent = evidence_top[0][:300]
        if not first_sent.endswith("."):
            first_sent += "."
        citation = f"{company}/{doc_type}/{fiscal_year}/MD&A/drivers"
        output = f"{output} [Source: {first_sent}] [{citation}]"

    return NarrativeAnswer(
        answered=True,
        output=output,
        drivers=drivers_human,
        evidence_sentences=evidence_top,
        confidence=min(0.85, 0.5 + 0.1 * len(drivers_human)),
        audit_trail={
            "subject":               subject,
            "n_causal_sentences":    len(causal_sentences),
            "driver_categories":     list(driver_hits.keys()),
            "n_drivers":             len(drivers_human),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Segment analysis — "which segment dragged growth?"
# ─────────────────────────────────────────────────────────────────────────────

# Common segment naming patterns (3M, AAPL, AMZN, etc.)
# FIX-v5 (2026-06-05): now COMPANY-SCOPED to prevent cross-company
# confusion (e.g. picking "Services" for 3M because Apple has Services)
COMPANY_SEGMENTS: Dict[str, List[str]] = {
    "3m": [
        "safety and industrial",
        "transportation and electronics",
        "health care",
        "healthcare",
        "consumer",
    ],
    "apple": [
        "iphone", "mac", "ipad", "wearables", "services",
        "americas", "europe", "greater china", "japan",
        "asia pacific",
    ],
    "amazon": [
        "amazon web services", "aws",
        "north america", "international",
        "online stores", "physical stores",
        "advertising services", "third-party seller services",
        "subscription services",
    ],
    "microsoft": [
        "productivity and business processes",
        "intelligent cloud", "more personal computing",
        "azure", "office", "linkedin", "dynamics",
        "gaming", "windows", "server products",
    ],
    "alphabet": [
        "google search", "youtube", "google network",
        "google cloud", "google other", "other bets",
    ],
    "google": [
        "google search", "youtube", "google network",
        "google cloud", "google other", "other bets",
    ],
    "nvidia": [
        "data center", "gaming", "professional visualization",
        "automotive", "oem",
    ],
    "tesla": [
        "automotive", "energy generation", "services and other",
    ],
    "intel": [
        "client computing", "data center", "network",
        "mobileye", "foundry services",
    ],
}

# Fallback list when company is unknown
SEGMENT_NAME_HINTS = sorted(
    {s for segs in COMPANY_SEGMENTS.values() for s in segs},
    key=len, reverse=True,
)


def extract_segment_answer(
    question: str,
    raw_text: str,
    bm25_results: List[Dict],
    company: str = "",
    fiscal_year: str = "",
    doc_type: str = "",
) -> NarrativeAnswer:
    """
    Handle "which segment dragged growth?" style questions.
    Best-effort: scans for segment names + growth/decline words.
    """
    if not detect_segment_question(question):
        return NarrativeAnswer(answered=False)

    corpus = _gather_search_text(raw_text, bm25_results, "segment")
    if not corpus:
        return NarrativeAnswer(
            answered=False,
            audit_trail={"reason": "empty corpus"},
        )

    q_lc = question.lower()
    looking_for_low = any(
        w in q_lc for w in
        ("dragged", "lowest", "weakest", "worst", "decline", "decreased", "fell")
    )

    # FIX-v5: Use company-scoped segment hints (prevents 3M from picking
    # "Services" which is Apple's/Amazon's segment).
    company_key = (company or "").lower().strip()
    company_key = company_key.split()[0] if company_key else ""
    # Map common variants
    company_aliases = {
        "3m": "3m", "mmm": "3m",
        "apple": "apple", "aapl": "apple",
        "amazon": "amazon", "amzn": "amazon",
        "microsoft": "microsoft", "msft": "microsoft",
        "google": "google", "googl": "google", "alphabet": "alphabet",
        "nvidia": "nvidia", "nvda": "nvidia",
        "tesla": "tesla", "tsla": "tesla",
        "intel": "intel", "intc": "intel",
    }
    resolved = company_aliases.get(company_key, "")
    if resolved and resolved in COMPANY_SEGMENTS:
        segment_hints = COMPANY_SEGMENTS[resolved]
        logger.info("[narrative_extractor] using %s-scoped segments (%d names)",
                    resolved, len(segment_hints))
    else:
        segment_hints = SEGMENT_NAME_HINTS
        logger.info("[narrative_extractor] unknown company '%s' — using all segment hints",
                    company_key)

    sentences = _split_sentences(corpus)

    # Find sentences with segment hints AND growth language
    candidates: List[Tuple[str, str, float]] = []   # (segment, sentence, percent_pp)
    for s in sentences:
        s_lc = s.lower()
        seg_name = None
        for hint in sorted(segment_hints, key=len, reverse=True):
            if hint in s_lc:
                seg_name = hint
                break
        if not seg_name:
            continue

        # Find a percent number nearby
        m_pct = re.search(r"(-?\d{1,3}(?:\.\d+)?)\s*%", s)
        pct = float(m_pct.group(1)) if m_pct else 0.0

        # Bias by direction
        if "declin" in s_lc or "fell" in s_lc or "decreas" in s_lc or "shrunk" in s_lc:
            pct = -abs(pct)

        candidates.append((seg_name, s, pct))

    if not candidates:
        return NarrativeAnswer(
            answered=False,
            audit_trail={"reason": "no segment + growth sentences found"},
        )

    # Pick the lowest if user asked for lowest/dragged
    if looking_for_low:
        candidates.sort(key=lambda x: x[2])
    else:
        candidates.sort(key=lambda x: -x[2])

    best_seg, best_sent, best_pct = candidates[0]
    sign = "" if best_pct >= 0 else "-"
    pct_str = f"{sign}{abs(best_pct):.1f}%" if best_pct != 0 else "n/a"

    direction = "shrunk" if best_pct < 0 else "grew"
    output = (
        f"The {best_seg.title()} segment {direction} by {pct_str}"
        f" — the {'lowest' if looking_for_low else 'highest'} of the segments. "
        f"[Source: {best_sent[:250]}]"
    )
    citation = f"{company}/{doc_type}/{fiscal_year}/segment_analysis"
    output += f" [{citation}]"

    return NarrativeAnswer(
        answered=True,
        output=output,
        drivers=[best_seg],
        evidence_sentences=[best_sent],
        confidence=0.7,
        audit_trail={
            "best_segment":  best_seg,
            "best_pct":      best_pct,
            "candidates":    len(candidates),
            "looking_for":   "lowest" if looking_for_low else "highest",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_q = "What drove operating margin change as of FY2022 for 3M?"
    test_text = (
        "Operating income margin was 19.4% compared to 22.2% last year. "
        "The decrease in operating margin was primarily driven by significant "
        "litigation charges related to Combat Arms earplug matters and Aearo "
        "Technologies, as well as the exit from PFAS manufacturing. "
        "Results were also impacted by divestiture activity and the exit from "
        "Russia following the geopolitical situation. "
        "Currency translation provided a partial offset."
    )

    print("\n[narrative_extractor] self-test 1 — what drove operating margin")
    ans = extract_drivers(test_q, test_text, [], "3M", "FY2022", "10-K")
    print(f"  answered:    {ans.answered}")
    print(f"  drivers:     {ans.drivers}")
    print(f"  confidence:  {ans.confidence}")
    print(f"  output:\n    {ans.output}")

    print("\n[narrative_extractor] self-test 2 — segment dragged growth")
    test_q2 = "If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?"
    test_text2 = (
        "Safety and Industrial revenue grew 4.5% organically. "
        "Transportation and Electronics revenue grew 2.1%. "
        "Health Care revenue grew 8.3%. "
        "Consumer revenue declined by 0.9% organically as demand softened. "
    )
    ans2 = extract_segment_answer(test_q2, test_text2, [], "3M", "FY2022", "10-K")
    print(f"  answered:    {ans2.answered}")
    print(f"  best:        {ans2.drivers}")
    print(f"  confidence:  {ans2.confidence}")
    print(f"  output:\n    {ans2.output}")
