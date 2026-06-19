"""
question_simplifier.py
Deterministic pre-processor that rewrites twisted FinanceBench questions into
simple canonical forms BEFORE the resolver/LLM sees them. No LLM, no VRAM.
Safe: if it can't simplify confidently, returns the original unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# ── Filler / hedge clauses to strip (zero information) ───────────────────────
_FILLER_PATTERNS = [
    # politeness / requests
    r"(?:could|can|would|will) you (?:please |kindly )?(?:tell me|explain|describe|determine|calculate|provide|give me|share|clarify|confirm|elaborate on|walk me through)",
    r"(?:please |kindly )?(?:help me |assist me in )?(?:understand|determine|calculate|find|identify|figure out|work out|assess|evaluate)",
    r"i(?:'d| would) (?:like|love) to (?:know|understand|see|learn)",
    r"i(?:'m| am) (?:curious|wondering|interested) (?:about|to know|whether|if)",
    r"let me know",
    # hedging / framing
    r"taking into (?:account|consideration) (?:the )?(?:figures|information|data|numbers|context|disclosures?)(?: (?:disclosed|provided|presented|reported|given|above))?,?",
    r"based (?:on|upon) (?:the )?(?:information|data|figures|context|financials?|disclosures?|statements?)(?: (?:provided|disclosed|given|presented|reported|above))?,?",
    r"(?:given|considering|in light of|in view of) (?:the )?(?:figures|information|data|context|financials?|disclosures?|above)?,?",
    r"with (?:reference|regard|respect) to (?:the )?(?:above|disclosed|provided|reported|figures|information|data|statements?)?,?",
    r"having regard to (?:the )?(?:above|disclosed|figures|information)?,?",
    r"(?:from|in) (?:the|a) (?:financial|reporting|analytical|investor's?|accounting) (?:standpoint|perspective|point of view|lens|angle),?",
    r"(?:in|from) (?:your|the|an?) (?:assessment|estimation|view|opinion|judgment|analysis|reading),?",
    r"all (?:things|factors) considered,?",
    r"as (?:per|disclosed in|stated in|reported in|shown in|seen in) (?:the )?(?:filing|report|statement|10-?k|10-?q|8-?k|document|disclosure|financials?)s?,?",
    # reasonableness wrappers
    r"would it be (?:reasonable|fair|accurate|appropriate|correct) to (?:say|state|characterize|conclude|assume|describe|consider|infer|claim|argue|suggest)(?: that)?",
    r"is it (?:reasonable|fair|accurate|appropriate|correct|safe) to (?:say|state|conclude|assume|infer|claim|argue|suggest)(?: that)?",
    r"(?:do|does|can) (?:the (?:figures|data|numbers)|this) (?:suggest|indicate|imply|show|demonstrate)(?: that)?",
    r"(?:to what extent|in what way|how far) (?:can we say|is it true)(?: that)?",
    # conditional framing
    r"if (?:we|you|one) (?:were to )?(?:look|consider|examine|analyze|assess|evaluate)(?: at)?,?",
    r"when (?:we|you|one) (?:look|consider|examine|analyze)(?: at)?,?",
    # verbose openers
    r"(?:i'?m trying to|i want to|i need to) (?:figure out|understand|determine|calculate|assess)",
]

# ── Verbose finance phrasings → canonical short metric phrasing ──────────────
_PHRASE_MAP = [
    # ── capital intensity ──
    (r"capital expenditures?\s+relative to\s+(?:its )?(?:total )?revenue(?:s| base)?", "capex to revenue ratio"),
    (r"(?:level of )?capital (?:expenditure|spending|investment)\s+(?:as a (?:proportion|percentage|share) of|versus|against|compared to)\s+(?:its )?revenue", "capex to revenue ratio"),
    (r"(?:indicative of|characteri[sz]ed? as|exhibit(?:s|ing)?|reflect(?:s|ing)?|suggest(?:s|ing)? a state of)\s+capital[- ]?intensit(?:y|ive)", "capital intensive"),
    (r"how (?:capital[- ]?intensive|much capital is required)", "capital intensity"),
    (r"capital[- ]?intensit(?:y|ive)\s+(?:nature|profile|characteristics?)", "capital intensity"),

    # ── liquidity ──
    (r"ability to (?:meet|cover|service|satisfy|pay)\s+(?:its )?(?:short[- ]?term )?(?:financial )?(?:obligations|liabilities|debts?|commitments)", "current ratio"),
    (r"(?:short[- ]?term )?liquidity (?:position|profile|strength|health|standing|situation)", "quick ratio"),
    (r"how (?:liquid|much liquidity (?:does it have|is available))", "liquidity"),
    (r"(?:can|is it able to)\s+cover\s+(?:its )?current liabilities", "current ratio"),

    # ── leverage / solvency ──
    (r"(?:degree|level|extent|amount) of (?:financial )?(?:leverage|gearing|indebtedness)", "debt to equity ratio"),
    (r"(?:reliance|dependence) (?:on|upon)\s+(?:debt|borrowing|leverage|external financing)", "debt to equity"),
    (r"(?:indebtedness|debt (?:load|burden|level))\s+relative to\s+(?:its )?(?:equity|assets)", "debt to equity ratio"),
    (r"how (?:leveraged|indebted|much debt)", "debt to equity"),

    # ── profitability / margins ──
    (r"profitability\s+(?:as measured by|in terms of|with respect to)\s+(?:its )?(?:net |operating |gross )?margin", "net margin"),
    (r"how profitable(?: (?:is|was) (?:the company|it))?", "net margin"),
    (r"(?:net |operating |gross )?(?:profit )?margin\s+(?:profile|performance|level|trend|trajectory)", "operating margin"),
    (r"(?:bottom[- ]?line|earnings) (?:performance|strength|profile)", "net income"),

    # ── growth ──
    (r"(?:year[- ]?over[- ]?year |yoy |annual )?(?:rate of |percentage |%? )?(?:growth|increase|change|movement|expansion) in\s+(?:its )?", "growth in "),
    (r"how (?:did|has|have)\s+(?:its )?(\w+)\s+(?:grow|grown|change[ds]?|increase[ds]?|move[ds]?|evolve[ds]?|develop(?:ed)?)", r"\1 growth"),
    (r"(?:trend|trajectory|direction) (?:in|of)\s+(?:its )?(\w+)\s+over (?:the )?(?:year|period)", r"\1 growth"),

    # ── cash flow ──
    (r"(?:its )?(?:ability to )?generat(?:e|ion of)\s+(?:free )?cash(?: flow)?", "free cash flow"),
    (r"cash (?:generation|conversion)\s+(?:profile|ability|capability|strength)", "free cash flow"),
    (r"(?:free )?cash flow\s+(?:generation|profile|performance)", "free cash flow"),

    # ── returns ──
    (r"return (?:generated )?(?:on|for)\s+(?:its )?(?:shareholders'?|stockholders'?)\s+equity", "return on equity"),
    (r"how (?:well|effectively) (?:does it|it)\s+(?:use|utili[sz]e|deploy)\s+(?:its )?(?:assets|capital)", "return on assets"),
    (r"(?:efficiency|effectiveness) (?:in|of) (?:using|deploying)\s+(?:its )?(?:assets|capital)", "return on assets"),

    # ── dividends ──
    (r"(?:how much|what (?:portion|share|fraction))\s+of\s+(?:its )?(?:earnings|profits?|income)\s+(?:is|are|was|were)\s+(?:paid|distributed|returned)\s+(?:out )?(?:as|in)\s+dividends?", "dividend payout ratio"),
    (r"dividend (?:distribution|payment)\s+(?:trend|policy|pattern|history)", "dividend trend"),

    # ── working capital / turnover ──
    (r"how (?:quickly|fast|efficiently) (?:does it|it)\s+(?:sell|move|convert)\s+(?:its )?inventory", "inventory turnover"),
    (r"inventory (?:management|conversion)\s+(?:efficiency|profile)", "inventory turnover"),
    (r"working capital\s+(?:position|management|profile|situation)", "working capital"),
]


@dataclass
class SimplifiedQuestion:
    original:   str
    simplified: str
    changed:    bool = False
    notes:      List[str] = field(default_factory=list)


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" ,.;:")


def simplify_question(question: str) -> SimplifiedQuestion:
    if not question or len(question) < 5:
        return SimplifiedQuestion(original=question, simplified=question)

    original = question
    q = " " + question.strip() + " "
    notes: List[str] = []

    for pat in _FILLER_PATTERNS:
        new = re.sub(pat, " ", q, flags=re.IGNORECASE)
        if new != q:
            notes.append("stripped filler")
            q = new

    for pat, repl in _PHRASE_MAP:
        new = re.sub(pat, repl, q, flags=re.IGNORECASE)
        if new != q:
            notes.append("mapped phrase")
            q = new
    q = _collapse_ws(q)

    if q and not q.endswith("?"):
        q = q + "?"
    if q:
        q = q[0].upper() + q[1:]

    changed = _collapse_ws(q).lower() != _collapse_ws(original).lower()
    if not q or len(q) < 6:
        return SimplifiedQuestion(original=original, simplified=original)

    return SimplifiedQuestion(original=original, simplified=q, changed=changed, notes=notes)


if __name__ == "__main__":
    tests = [
        "Taking into account the figures disclosed, would it be reasonable to "
        "characterize the company's capital expenditure relative to its revenue "
        "base as indicative of capital intensity?",
        "Based on the information provided, could you please determine how "
        "profitable the company was in FY2022?",
        "With reference to the above figures, is it fair to say the company has "
        "a strong short-term liquidity position?",
        "Considering the disclosures, how did its revenue change year-over-year?",
        "What was 3M's revenue in FY2018?",   # already simple — must stay intact
    ]
    for t in tests:
        r = simplify_question(t)
        print("\nORIG:", t[:70], "...")
        print("SIMP:", r.simplified)
        print("changed:", r.changed)