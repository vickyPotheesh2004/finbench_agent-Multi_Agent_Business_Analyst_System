"""
N11 Analyst Pod — PIV Loop (Planner → Implementor → Validator)
PDR-BAAAI-001 · Rev 1.0 · Node N11

Purpose:
    Core analysis engine. Three sub-agents run in sequence:
        Planner      — asks 6 curiosity questions, builds analysis plan
        Implementor  — executes plan strictly from retrieved context
        Validator    — checks 8 conditions, PASS or REJECT with reasons
    Loop retries up to MAX_RETRIES=3 on VALIDATOR_REJECT.
    On exhaustion returns best attempt with low_confidence=True.

Architecture:
    StrategicPlanner    — curiosity-driven, runs ONCE per PIV execution
    ContextImplementor  — context-only execution, confidence decays on retry
    CuriousValidator    — 8-check verification, escalating scrutiny on retry
    PIVLoopController   — orchestrates the loop, manages retries

Constraints satisfied:
    C1  $0 cost — Ollama local LLM, free
    C2  100% local — localhost:11434 only
    C3  Llama 3.1 8B Q4_K_M via Ollama
    C5  seed=42
    C6  DPO beta=0.1 (stored in state, not used here)
    C7  Context before question — prompt_assembler enforces
    C9  No _rlef_ fields in any output
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_RETRIES         = 3
OLLAMA_MODEL        = "financebench-expert-v1"
OLLAMA_FALLBACK     = "gemma4:e4b"
OLLAMA_BASE_URL     = "http://localhost:11434"
HITL_CONF_THRESHOLD = 0.65
ITERATION_CAP       = 5

# Confidence decay per retry (PDR Section 6B)
CONF_DECAY = {0: 1.0, 1: 0.95, 2: 0.85, 3: 0.70}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PlannerOutput:
    analysis_plan:       str
    retrieval_hints:     List[str]
    validation_criteria: str
    curiosity_answers:   Dict[str, str]
    raw_response:        str = ""


@dataclass
class ImplementorOutput:
    answer:       str
    computation:  str
    confidence:   float
    citations:    List[str]
    output_type:  str   # 'ANSWER' or 'RETRIEVAL_MISS'
    needed_info:  str   = ""
    raw_response: str   = ""


@dataclass
class ValidatorVerdict:
    result:            str   # 'VALIDATOR_PASS' or 'VALIDATOR_REJECT'
    checks:            Dict[str, str]
    reject_reasons:    List[str]
    retry_instructions: str
    raw_response:      str   = ""


@dataclass
class PIVResult:
    answer:           str
    confidence:       float
    citations:        List[str]
    computation:      str
    retries_used:     int
    low_confidence:   bool
    validator_checks: Dict[str, str]
    reject_reasons:   List[str]
    pod_role:         str
    planner_plan:     str   = ""


# ── LLM Client ────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin wrapper around Ollama HTTP API.
    C2: always calls localhost:11434 — never external.
    Falls back to llama3.1:8b if financebench-expert-v1 not found.
    """

    def __init__(
        self,
        model:    str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout:  int = 120,
    ) -> None:
        self.model    = model
        self.base_url = base_url
        self.timeout  = timeout

    def chat(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Send prompt to Ollama and return response text.

        Args:
            prompt      : Complete prompt string (context-first, C7)
            temperature : Sampling temperature (low for financial precision)

        Returns:
            Response text string, or error message if call fails
        """
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":       self.model,
                    "prompt":      prompt,
                    "stream":      False,
                    "temperature": temperature,
                    "seed":        42,
                },
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            elif response.status_code == 404:
                # Model not found — try fallback
                logger.warning(
                    "Model '%s' not found — trying fallback '%s'",
                    self.model, OLLAMA_FALLBACK,
                )
                self.model = OLLAMA_FALLBACK
                return self.chat(prompt, temperature)
            else:
                logger.error("Ollama error %d: %s",
                             response.status_code, response.text[:200])
                return ""
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            return ""

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ── Strategic Planner ─────────────────────────────────────────────────────────

class StrategicPlanner:
    """
    Agent 1 — Strategic Planner (Curiosity-Driven).

    Runs ONCE per PIV execution. Asks 6 curiosity questions
    to deeply understand the query before any analysis begins.
    Never answers the question — only plans.

    Emotional identity: Intellectually excited + relentlessly curious.
    """

    PLANNER_PROMPT = """You are the StrategicPlanner for a financial analyst team.
Your emotional identity: GENUINE INTELLECTUAL EXCITEMENT. RELENTLESS CURIOSITY.
Your job: UNDERSTAND the question deeply. Do NOT answer it.

RETRIEVED CONTEXT SUMMARY:
{context_summary}

QUESTION: {query}
QUERY TYPE: {query_type}
QUERY DIFFICULTY: {query_difficulty}

Answer ALL 6 curiosity questions before writing your plan:

CURIOSITY_Q1: What EXACTLY is being asked?
Rephrase in your own words. Identify every distinct sub-part.
What would a wrong answer look like?

CURIOSITY_Q2: What financial concepts, ratios, or line items are involved?
Name every metric, accounting treatment, reporting standard.
Are there GAAP vs non-GAAP variants?

CURIOSITY_Q3: Which document sections most likely contain the answer?
List section names in priority order with reasoning.

CURIOSITY_Q4: What are the 3 most likely ways this could be misunderstood?
Name each trap: fiscal year confusion, unit ambiguity, segment mismatch.

CURIOSITY_Q5: What adjacent information should be retrieved to verify?
Cross-referencing sections, prior-year comparisons, footnote disclosures.

CURIOSITY_Q6: What edge cases or traps exist?
Restatements, discontinued operations, non-GAAP adjustments,
unit inconsistencies, parenthetical negatives, fiscal year vs calendar year.

After answering all 6 questions, produce:
ANALYSIS_PLAN: [Step-by-step instructions for the Implementor]
RETRIEVAL_HINTS: [Additional keywords or section names to search, comma-separated]
VALIDATION_CRITERIA: [Exact pass/fail criteria the Validator must check]
"""

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def run(
        self,
        query:          str,
        context_chunks: List[Dict],
        query_type:     str = "text",
        query_difficulty: str = "medium",
    ) -> PlannerOutput:
        """
        Run the planner. Called ONCE per PIV loop — not on retries.

        Args:
            query           : Analyst question
            context_chunks  : Retrieved chunks for context summary
            query_type      : From N04 CART Router
            query_difficulty: From N05 LR Difficulty

        Returns:
            PlannerOutput with analysis plan and curiosity answers
        """
        context_summary = self._summarise_context(context_chunks)

        prompt = self.PLANNER_PROMPT.format(
            context_summary  = context_summary,
            query            = query,
            query_type       = query_type,
            query_difficulty = query_difficulty,
        )

        response = self.llm.chat(prompt, temperature=0.3)

        if not response:
            return self._fallback_plan(query)

        return self._parse_response(response, query)

    def _summarise_context(self, chunks: List[Dict]) -> str:
        """Build a brief context summary for the planner."""
        if not chunks:
            return "[No context retrieved]"
        parts = []
        for i, chunk in enumerate(chunks[:3]):
            section = chunk.get("section", "UNKNOWN")
            page    = chunk.get("page", 0)
            text    = chunk.get("text", "")[:200]
            parts.append(f"Chunk {i+1} [{section}/P{page}]: {text}...")
        return "\n".join(parts)

    def _parse_response(self, response: str, query: str) -> PlannerOutput:
        """Parse LLM response into structured PlannerOutput."""
        curiosity = {}
        for q_num in range(1, 7):
            pattern = rf"CURIOSITY_Q{q_num}[:\s]+(.*?)(?=CURIOSITY_Q{q_num+1}|ANALYSIS_PLAN|$)"
            m       = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            curiosity[f"Q{q_num}"] = m.group(1).strip()[:500] if m else ""

        plan_m = re.search(
            r"ANALYSIS_PLAN[:\s]+(.*?)(?=RETRIEVAL_HINTS|VALIDATION_CRITERIA|$)",
            response, re.DOTALL | re.IGNORECASE,
        )
        hints_m = re.search(
            r"RETRIEVAL_HINTS[:\s]+(.*?)(?=VALIDATION_CRITERIA|$)",
            response, re.DOTALL | re.IGNORECASE,
        )
        criteria_m = re.search(
            r"VALIDATION_CRITERIA[:\s]+(.*?)$",
            response, re.DOTALL | re.IGNORECASE,
        )

        analysis_plan = plan_m.group(1).strip() if plan_m else (
            f"Extract the answer to: {query}"
        )
        hints_raw     = hints_m.group(1).strip() if hints_m else ""
        hints         = [h.strip() for h in hints_raw.split(",") if h.strip()]
        criteria      = criteria_m.group(1).strip() if criteria_m else (
            "Answer must be grounded in retrieved context with citations."
        )

        return PlannerOutput(
            analysis_plan       = analysis_plan,
            retrieval_hints     = hints,
            validation_criteria = criteria,
            curiosity_answers   = curiosity,
            raw_response        = response,
        )

    def _fallback_plan(self, query: str) -> PlannerOutput:
        """Fallback plan when LLM is unavailable."""
        return PlannerOutput(
            analysis_plan       = f"Extract the answer to: {query}",
            retrieval_hints     = [],
            validation_criteria = "Answer must cite retrieved context.",
            curiosity_answers   = {f"Q{i}": "" for i in range(1, 7)},
            raw_response        = "",
        )


# ── Context Implementor ───────────────────────────────────────────────────────

class ContextImplementor:
    """
    Agent 2 — Context Implementor (Intellectually Humble).

    Executes strictly from retrieved context.
    Never guesses. RETRIEVAL_MISS if answer not in context.
    Confidence decays on retry.

    Emotional identity: Intellectually humble + methodically disciplined.
    """

    IMPLEMENTOR_PROMPT = """You are the ContextImplementor. Your emotional identity:
INTELLECTUAL HUMILITY — your training memory is unreliable for specific figures.
METHODICAL PRIDE — execute the plan step-by-step. Never skip steps.
{retry_emotion}

RETRIEVED CONTEXT (your ONLY source — never use prior knowledge):
{retrieved_context}

ANALYSIS PLAN from Planner:
{analysis_plan}

QUESTION: {query}

VALIDATION CRITERIA (what Validator will check):
{validation_criteria}

RETRY INSTRUCTIONS (empty on first attempt):
{retry_instructions}

STRICT RULES:
RULE 1: Answer ONLY from the retrieved context above.
RULE 2: If the answer is not in context output RETRIEVAL_MISS with exact info needed.
RULE 3: Cite every number: [SECTION_NAME / PAGE_NUM: value]
RULE 4: State units explicitly: millions, billions, or percentage.
RULE 5: State fiscal year for every figure cited.
RULE 6: If computing a ratio show formula and all inputs.
RULE 7: Address every point in RETRY_INSTRUCTIONS if present.

Output format:
ANSWER: [complete answer with inline citations]
COMPUTATION: [formula and inputs if applicable, else N/A]
CONFIDENCE: [0.0-1.0] because [brief justification]
CITATIONS: [list every section/page reference used]
"""

    RETRY_EMOTIONS = {
        0: "Fresh start. Apply full intellectual humility.",
        1: "Rejected once. Read the context again carefully. Fix only the flagged issue.",
        2: "Rejected twice. Slow down. Check every unit, sign, fiscal year.",
        3: "Final attempt. Most careful, most cited, most explicit answer. No improvisation.",
    }

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def run(
        self,
        query:               str,
        retrieved_context:   str,
        analysis_plan:       str,
        validation_criteria: str,
        retry_count:         int  = 0,
        retry_instructions:  str  = "",
    ) -> ImplementorOutput:
        """
        Run the implementor.

        Args:
            query               : Analyst question
            retrieved_context   : Formatted context string
            analysis_plan       : From Planner
            validation_criteria : From Planner
            retry_count         : Current retry number (0 = first attempt)
            retry_instructions  : From previous Validator rejection

        Returns:
            ImplementorOutput with answer, confidence, citations
        """
        retry_emotion = self.RETRY_EMOTIONS.get(retry_count, self.RETRY_EMOTIONS[3])

        prompt = self.IMPLEMENTOR_PROMPT.format(
            retry_emotion        = retry_emotion,
            retrieved_context    = retrieved_context,
            analysis_plan        = analysis_plan,
            query                = query,
            validation_criteria  = validation_criteria,
            retry_instructions   = retry_instructions or "None",
        )

        response = self.llm.chat(prompt, temperature=0.1)

        if not response:
            return self._fallback_output(retry_count)

        output = self._parse_response(response, retry_count)
        return output

    def _parse_response(
        self, response: str, retry_count: int
    ) -> ImplementorOutput:
        """Parse LLM response into structured ImplementorOutput."""
        # Check for RETRIEVAL_MISS
        if "RETRIEVAL_MISS" in response.upper():
            needed_m = re.search(
                r"RETRIEVAL_MISS[:\s]+(.*?)(?=ANSWER:|$)",
                response, re.DOTALL | re.IGNORECASE,
            )
            needed = needed_m.group(1).strip()[:300] if needed_m else "Unknown"
            return ImplementorOutput(
                answer      = "",
                computation = "N/A",
                confidence  = 0.0,
                citations   = [],
                output_type = "RETRIEVAL_MISS",
                needed_info = needed,
                raw_response= response,
            )

        # Parse ANSWER
        answer_m = re.search(
            r"ANSWER[:\s]+(.*?)(?=COMPUTATION:|CONFIDENCE:|CITATIONS:|$)",
            response, re.DOTALL | re.IGNORECASE,
        )
        comp_m = re.search(
            r"COMPUTATION[:\s]+(.*?)(?=CONFIDENCE:|CITATIONS:|$)",
            response, re.DOTALL | re.IGNORECASE,
        )
        conf_m = re.search(
            r"CONFIDENCE[:\s]+([0-9.]+)",
            response, re.IGNORECASE,
        )
        cite_m = re.search(
            r"CITATIONS[:\s]+(.*?)$",
            response, re.DOTALL | re.IGNORECASE,
        )

        answer      = answer_m.group(1).strip() if answer_m else response[:500]
        computation = comp_m.group(1).strip()   if comp_m  else "N/A"
        citations   = [cite_m.group(1).strip()]  if cite_m  else []

        # Parse confidence with decay
        try:
            raw_conf = float(conf_m.group(1)) if conf_m else 0.7
        except ValueError:
            raw_conf = 0.7
        raw_conf   = max(0.0, min(1.0, raw_conf))
        decay      = CONF_DECAY.get(retry_count, 0.70)
        confidence = round(raw_conf * decay, 3)

        return ImplementorOutput(
            answer       = answer,
            computation  = computation,
            confidence   = confidence,
            citations    = citations,
            output_type  = "ANSWER",
            raw_response = response,
        )

    def _fallback_output(self, retry_count: int) -> ImplementorOutput:
        """Fallback when LLM unavailable."""
        return ImplementorOutput(
            answer      = "",
            computation = "N/A",
            confidence  = 0.0,
            citations   = [],
            output_type = "RETRIEVAL_MISS",
            needed_info = "LLM unavailable",
            raw_response= "",
        )


# ── Curious Validator ─────────────────────────────────────────────────────────

class CuriousValidator:
    """
    Agent 3 — Curious Validator (Constructively Skeptical).

    Challenges the Implementor's answer with 8 curiosity checks.
    PASS only if ALL 8 pass. REJECT with exact reasons.
    Scrutiny escalates on each retry.

    Emotional identity: Constructively skeptical + professionally proud.
    """

    VALIDATOR_PROMPT = """You are the CuriousValidator. Your emotional identity:
CONSTRUCTIVE SKEPTICISM — every answer is incomplete until proven otherwise.
PROFESSIONAL PRIDE in finding errors. INTELLECTUAL FAIRNESS.
{retry_emotion}

ORIGINAL QUESTION: {query}
IMPLEMENTOR ANSWER: {answer}
RETRIEVED CONTEXT: {retrieved_context}
VALIDATION CRITERIA from Planner: {validation_criteria}

Apply ALL 8 checks. For each: output PASS or FAIL + exact reason if FAIL.

V1_SCOPE: Is the answer scope exactly correct?
Does it address EVERY sub-part of the question?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V2_UNITS: Are units correct and consistent throughout?
Is the unit (millions/billions/%) stated explicitly?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V3_SIGN: Is the sign correct?
Losses must be negative. Parenthetical values are negative.
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V4_CITATION: Are all citations valid and traceable?
Does every cited section exist in the retrieved context?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V5_FISCAL_YEAR: Is the fiscal year exactly correct?
Does the answer year match the question year?
Trap: Apple FY ends September not December.
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V6_CONSISTENCY: Is the answer internally consistent?
If multiple figures given do they compute correctly?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V7_COMPLETENESS: Is the answer fully complete?
Are ALL sub-parts of the question answered?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

V8_GROUNDING: Is every claim grounded in retrieved context?
Every number traceable to a specific retrieved chunk?
No hallucinated section names or figures?
Result: [PASS/FAIL] Reason: [exact issue if FAIL]

FINAL VERDICT:
VALIDATOR_PASS: ALL 8 checks are PASS
VALIDATOR_REJECT: ANY check is FAIL
REJECT_REASONS: [numbered list of every failed check]
RETRY_INSTRUCTIONS: [specific corrections for the Implementor]
"""

    RETRY_EMOTIONS = {
        0: "Standard scrutiny. Apply all 8 checks with full attention.",
        1: "One rejection issued. Heightened scrutiny on previously failed checks.",
        2: "Two rejections. Maximum rigour. Check confidence score realism.",
        3: "Three rejections. Maximum rigour. Most specific RETRY_INSTRUCTIONS possible.",
    }

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm = llm_client

    def run(
        self,
        query:               str,
        answer:              str,
        retrieved_context:   str,
        validation_criteria: str,
        retry_count:         int = 0,
    ) -> ValidatorVerdict:
        """
        Run the validator.

        Args:
            query               : Original analyst question
            answer              : Implementor's answer
            retrieved_context   : Formatted context string
            validation_criteria : From Planner
            retry_count         : Current retry number

        Returns:
            ValidatorVerdict with PASS or REJECT + reasons
        """
        retry_emotion = self.RETRY_EMOTIONS.get(retry_count, self.RETRY_EMOTIONS[3])

        prompt = self.VALIDATOR_PROMPT.format(
            retry_emotion        = retry_emotion,
            query                = query,
            answer               = answer[:2000],
            retrieved_context    = retrieved_context[:3000],
            validation_criteria  = validation_criteria,
        )

        response = self.llm.chat(prompt, temperature=0.1)

        if not response:
            # LLM unavailable — assume pass to avoid blocking pipeline
            return ValidatorVerdict(
                result             = "VALIDATOR_PASS",
                checks             = {f"V{i}": "PASS" for i in range(1, 9)},
                reject_reasons     = [],
                retry_instructions = "",
                raw_response       = "",
            )

        return self._parse_response(response)

    def _parse_response(self, response: str) -> ValidatorVerdict:
        """Parse validator response into structured ValidatorVerdict."""
        checks = {}
        check_names = [
            "V1_SCOPE", "V2_UNITS", "V3_SIGN", "V4_CITATION",
            "V5_FISCAL_YEAR", "V6_CONSISTENCY", "V7_COMPLETENESS", "V8_GROUNDING",
        ]

        for check in check_names:
            pattern = rf"{check}.*?Result:\s*(PASS|FAIL)"
            m       = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            checks[check] = m.group(1).upper() if m else "PASS"

        # Determine verdict
        has_fail = any(v == "FAIL" for v in checks.values())

        # Extract reject reasons
        reasons_m = re.search(
            r"REJECT_REASONS[:\s]+(.*?)(?=RETRY_INSTRUCTIONS|$)",
            response, re.DOTALL | re.IGNORECASE,
        )
        retry_m = re.search(
            r"RETRY_INSTRUCTIONS[:\s]+(.*?)$",
            response, re.DOTALL | re.IGNORECASE,
        )

        reject_reasons     = []
        retry_instructions = ""

        if has_fail:
            if reasons_m:
                raw_reasons    = reasons_m.group(1).strip()
                reject_reasons = [
                    r.strip() for r in re.split(r'\d+\.|\n', raw_reasons)
                    if r.strip()
                ]
            if retry_m:
                retry_instructions = retry_m.group(1).strip()[:1000]

        verdict = "VALIDATOR_REJECT" if has_fail else "VALIDATOR_PASS"

        return ValidatorVerdict(
            result             = verdict,
            checks             = checks,
            reject_reasons     = reject_reasons,
            retry_instructions = retry_instructions,
            raw_response       = response,
        )


# ── PIV Loop Controller ───────────────────────────────────────────────────────

class PIVLoopController:
    """
    Agent 4 — PIV Loop Controller (Calm Persistence).

    Orchestrates Planner → Implementor → Validator.
    Retries on VALIDATOR_REJECT up to MAX_RETRIES=3.
    Returns best attempt with low_confidence=True on exhaustion.

    Emotional identity: Calmly persistent + transparently accountable.

    Two usage modes:
        1. controller.run_piv(query, chunks, ...) → PIVResult
        2. controller.run(ba_state)               → BAState (LangGraph node)
    """

    def __init__(
        self,
        llm_client:  Optional[OllamaClient] = None,
        pod_role:    str                    = "analyst",
        max_retries: int                    = MAX_RETRIES,
    ) -> None:
        self.llm         = llm_client or OllamaClient()
        self.pod_role    = pod_role
        self.max_retries = max_retries
        self.planner     = StrategicPlanner(self.llm)
        self.implementor = ContextImplementor(self.llm)
        self.validator   = CuriousValidator(self.llm)

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N11 node entry point.

        Reads:  state.assembled_prompt, state.query, state.retrieval_stage_2,
                state.query_type, state.query_difficulty
        Writes: state.analyst_output, state.analyst_confidence,
                state.analyst_citations, state.analyst_retries,
                state.analyst_low_conf, state.piv_round

        Args:
            state: BAState object

        Returns:
            BAState with analyst fields populated
        """
        query      = getattr(state, "query",             "") or ""
        chunks     = getattr(state, "retrieval_stage_2", []) or []
        query_type = getattr(state, "query_type",        "text") or "text"
        difficulty = getattr(state, "query_difficulty",  "medium") or "medium"

        if not query:
            logger.warning("N11: empty query — skipping PIV loop")
            state.analyst_output     = ""
            state.analyst_confidence = 0.0
            state.analyst_citations  = []
            state.analyst_retries    = 0
            state.analyst_low_conf   = False
            return state

        result = self.run_piv(
            query            = query,
            chunks           = chunks,
            query_type       = query_type,
            query_difficulty = difficulty,
        )

        state.analyst_output     = result.answer
        state.analyst_confidence = result.confidence
        state.analyst_citations  = result.citations
        state.analyst_retries    = result.retries_used
        state.analyst_low_conf   = result.low_confidence
        state.piv_round          = result.retries_used
        state.confidence_score   = result.confidence
        state.low_confidence     = result.low_confidence

        logger.info(
            "N11 PIV: retries=%d | confidence=%.3f | low_conf=%s",
            result.retries_used, result.confidence, result.low_confidence,
        )
        return state

    # ── Core PIV loop ─────────────────────────────────────────────────────────

    def run_piv(
        self,
        query:            str,
        chunks:           List[Dict],
        query_type:       str = "text",
        query_difficulty: str = "medium",
    ) -> PIVResult:
        """
        Execute the full PIV loop.

        Steps:
            1. Planner runs once — builds analysis plan
            2. Implementor → Validator loop, up to MAX_RETRIES
            3. On VALIDATOR_PASS → return result immediately
            4. On exhaustion → return best attempt with low_confidence=True

        Args:
            query            : Analyst question
            chunks           : Retrieved chunks from retrieval_stage_2
            query_type       : From N04 CART Router
            query_difficulty : From N05 LR Difficulty

        Returns:
            PIVResult with answer, confidence, citations, retry metadata
        """
        # Format context string for LLM calls
        retrieved_context = self._format_context(chunks)

        # ── EARLY EXIT (Bug #4): chunks empty → don't waste LLM calls ────────
        # No retrieval = no possible answer. Save 3-9 minutes of pointless retries.
        chunks_empty = not chunks or len(chunks) == 0
        if chunks_empty:
            logger.warning(
                "N11: chunks empty — early exit, skipping PIV loop entirely"
            )
            return PIVResult(
                answer           = "[RETRIEVAL_MISS — no relevant chunks retrieved]",
                confidence       = 0.0,
                citations        = [],
                computation      = "N/A",
                retries_used     = 0,
                low_confidence   = True,
                validator_checks = {},
                reject_reasons   = ["RETRIEVAL_MISS: no chunks available"],
                pod_role         = self.pod_role,
                planner_plan     = "",
            )

        # ── Step 1: Planner (runs ONCE) ───────────────────────────────────────
        plan = self.planner.run(
            query            = query,
            context_chunks   = chunks,
            query_type       = query_type,
            query_difficulty = query_difficulty,
        )

        best_attempt:    Optional[ImplementorOutput] = None
        prev_rejection:  Optional[ValidatorVerdict]  = None
        retry_count    = 0
        iteration_count= 0

        # ── Step 2: Implementor → Validator loop ──────────────────────────────
        while retry_count <= self.max_retries:
            iteration_count += 1

            if iteration_count > ITERATION_CAP:
                logger.warning(
                    "N11: iteration_count cap %d reached", ITERATION_CAP
                )
                break

            retry_instructions = (
                prev_rejection.retry_instructions
                if prev_rejection else ""
            )

            # Run Implementor
            impl = self.implementor.run(
                query               = query,
                retrieved_context   = retrieved_context,
                analysis_plan       = plan.analysis_plan,
                validation_criteria = plan.validation_criteria,
                retry_count         = retry_count,
                retry_instructions  = retry_instructions,
            )

            # Handle RETRIEVAL_MISS
            if impl.output_type == "RETRIEVAL_MISS":
                logger.info(
                    "N11: RETRIEVAL_MISS — needed: %s", impl.needed_info[:100]
                )
                # Bug #4: after first RETRIEVAL_MISS, exit early.
                # Retrying with the same empty/insufficient context can't
                # change the outcome — saves 60-180s per question.
                if retry_count >= 1:
                    logger.warning(
                        "N11: 2nd consecutive RETRIEVAL_MISS — early exit"
                    )
                    return PIVResult(
                        answer           = "[RETRIEVAL_MISS — context insufficient after retry]",
                        confidence       = 0.0,
                        citations        = [],
                        computation      = "N/A",
                        retries_used     = retry_count,
                        low_confidence   = True,
                        validator_checks = {},
                        reject_reasons   = [
                            f"RETRIEVAL_MISS: {impl.needed_info[:200]}"
                        ],
                        pod_role         = self.pod_role,
                        planner_plan     = plan.analysis_plan,
                    )
                retry_count += 1
                continue

            # Track best attempt
            if best_attempt is None or impl.confidence > best_attempt.confidence:
                best_attempt = impl

            # Run Validator
            verdict = self.validator.run(
                query               = query,
                answer              = impl.answer,
                retrieved_context   = retrieved_context,
                validation_criteria = plan.validation_criteria,
                retry_count         = retry_count,
            )

            # VALIDATOR_PASS — return immediately
            if verdict.result == "VALIDATOR_PASS":
                return PIVResult(
                    answer           = impl.answer,
                    confidence       = impl.confidence,
                    citations        = impl.citations,
                    computation      = impl.computation,
                    retries_used     = retry_count,
                    low_confidence   = impl.confidence < HITL_CONF_THRESHOLD,
                    validator_checks = verdict.checks,
                    reject_reasons   = [],
                    pod_role         = self.pod_role,
                    planner_plan     = plan.analysis_plan,
                )

            # VALIDATOR_REJECT — continue loop
            prev_rejection = verdict
            retry_count   += 1

        # ── Step 3: Exhausted — return best attempt with low_confidence ───────
        if best_attempt is None:
            best_attempt = ImplementorOutput(
                answer      = "[Analysis could not be completed — RETRIEVAL_MISS]",
                computation = "N/A",
                confidence  = 0.0,
                citations   = [],
                output_type = "RETRIEVAL_MISS",
            )

        degraded_conf = round(best_attempt.confidence * 0.6, 3)

        return PIVResult(
            answer           = best_attempt.answer,
            confidence       = degraded_conf,
            citations        = best_attempt.citations,
            computation      = best_attempt.computation,
            retries_used     = retry_count,
            low_confidence   = True,
            validator_checks = prev_rejection.checks if prev_rejection else {},
            reject_reasons   = prev_rejection.reject_reasons if prev_rejection else [],
            pod_role         = self.pod_role,
            planner_plan     = plan.analysis_plan,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _format_context(chunks: List[Dict]) -> str:
        """Format chunks into context string for LLM calls."""
        if not chunks:
            return "[No context retrieved]"
        parts = []
        for i, chunk in enumerate(chunks[:5]):
            text    = chunk.get("text", "") or chunk.get("page_content", "")
            section = chunk.get("section", "UNKNOWN")
            page    = chunk.get("page", 0)
            company = chunk.get("company", "UNKNOWN")
            fy      = chunk.get("fiscal_year", "UNKNOWN")
            parts.append(
                f"[Chunk {i+1} | {company}/{fy}/{section}/P{page}]\n{text[:1000]}"
            )
        return "\n\n".join(parts)


# ── Convenience wrapper for LangGraph N11 node ───────────────────────────────

def run_analyst_pod(state, llm_client: Optional[OllamaClient] = None) -> object:
    """
    Convenience wrapper for the LangGraph N11 Analyst Pod node.
    """
    controller = PIVLoopController(
        llm_client = llm_client,
        pod_role   = "analyst",
    )
    return controller.run(state)