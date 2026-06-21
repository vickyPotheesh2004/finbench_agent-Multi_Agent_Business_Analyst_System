from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

import warnings
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Disable GPU embedding contention with Ollama
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import gc
import json
import logging
import re
import sys
import time

from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

sys.path.insert(
    0,
    str(ROOT),
)

from src.pipeline.pipeline import (
    FinBenchPipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(
    "financebench_eval"
)

logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Paths
# ─────────────────────────────────────────────────────────────────────────────

FB_ROOT = (
    ROOT
    / "financebench_dataset"
    / "financebench"
)

FB_QUESTIONS = (
    FB_ROOT
    / "data"
    / "financebench_open_source.jsonl"
)

FB_METADATA = (
    FB_ROOT
    / "data"
    / "financebench_document_information.jsonl"
)

FB_PDFS_DIR = (
    FB_ROOT
    / "pdfs"
)

# ─────────────────────────────────────────────────────────────────────────────
# Results Directory
# ─────────────────────────────────────────────────────────────────────────────

if os.path.exists(
    "/content/drive/MyDrive"
):

    RESULTS_DIR = Path(
        "/content/drive/MyDrive/finbench_results"
    )

else:

    RESULTS_DIR = (
        ROOT
        / "eval"
        / "results"
    )

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def validate_financebench():

    required = [
        FB_QUESTIONS,
        FB_METADATA,
        FB_PDFS_DIR,
    ]

    missing = [
        str(p)
        for p in required
        if not p.exists()
    ]

    if missing:

        print()
        print("=" * 80)
        print(
            "❌ FINANCEBENCH DATASET MISSING"
        )
        print("=" * 80)
        print()

        for item in missing:

            print(f" - {item}")

        print()

        print(
            "Clone using:"
        )

        print(
            "git clone "
            "https://github.com/patronus-ai/financebench.git "
            "financebench_dataset"
        )

        raise FileNotFoundError(
            "FinanceBench dataset missing."
        )

    logger.info(
        "FinanceBench dataset validated"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def normalize_number(
    text: str,
):

    if not text:
        return ""

    s = (
        str(text)
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
    )

    s = re.sub(
        r"\s+(million|billion|thousand|bn|mn|m)\b",
        "",
        s,
        flags=re.I,
    )

    nums = re.findall(
        r"-?\d+\.?\d*",
        s,
    )

    if not nums:
        return ""

    try:

        value = float(nums[0])

        return (
            f"{value:.4f}"
            .rstrip("0")
            .rstrip(".")
        )

    except Exception:

        return ""


def numbers_match(
    predicted: str,
    gold: str,
    tolerance: float = 0.02,
):

    p = normalize_number(
        predicted
    )

    g = normalize_number(
        gold
    )

    if not p or not g:
        return False

    try:

        pf = float(p)

        gf = float(g)

        if abs(gf) < 1e-9:

            return abs(pf) < 1e-3

        return (
            abs(pf - gf)
            / abs(gf)
        ) < tolerance

    except Exception:

        return False


def overlap_score(
    predicted: str,
    gold: str,
):

    p = set(
        re.findall(
            r"\w+",
            predicted.lower(),
        )
    )

    g = set(
        re.findall(
            r"\w+",
            gold.lower(),
        )
    )

    if not p or not g:
        return 0.0

    return len(p & g) / len(p | g)


# MOVE-7 (2026-06-13): abstention / non-answer guard.
# The LLM often replies "it is not possible to determine ..." or "no explicit
# information ...". The old grader then scored these against a real gold answer
# via word-overlap and handed out FALSE positives (e.g. a "can't determine"
# reply matched gold "the quick ratio was 0.96"). A non-answer must NEVER count
# as correct, and RETRIEVAL_MISS likewise. This makes the score HONEST.
_ABSTAIN_MARKERS = (
    "not possible to determine", "cannot be determined",
    "cannot determine", "can't determine", "unable to determine",
    "no explicit", "not explicitly", "no information",
    "no specific information", "not enough information",
    "insufficient information", "not provided", "is unclear",
    "it is unclear", "cannot be calculated", "cannot calculate",
    "not available", "no mention", "not mentioned",
    "does not provide", "is not provided", "not specified",
    "retrieval_miss", "context insufficient",
)


def _is_non_answer(predicted: str) -> bool:
    """True if the prediction is an abstention / 'I don't know' style reply."""
    if not predicted:
        return True
    pl = predicted.lower()
    return any(marker in pl for marker in _ABSTAIN_MARKERS)


def is_correct(
    predicted: str,
    gold: str,
):

    if not predicted:

        return (
            False,
            "empty",
            0.0,
        )

    # MOVE-7: a non-answer can never be correct. Guard runs BEFORE overlap
    # scoring so "it is not possible to determine..." stops stealing credit
    # from a gold answer that contains a real value. EXCEPTION: if the gold
    # answer is ITSELF an abstention (rare), fall through to overlap so a
    # genuine "cannot be determined" match still counts.
    if _is_non_answer(predicted) and not _is_non_answer(gold):
        return (
            False,
            "non_answer",
            0.0,
        )

    pred_num = normalize_number(
        predicted
    )

    gold_num = normalize_number(
        gold
    )

    if pred_num and gold_num:

        if numbers_match(
            pred_num,
            gold_num,
        ):

            return (
                True,
                "numeric",
                1.0,
            )

        if numbers_match(
            pred_num,
            gold_num,
            tolerance=0.05,
        ):

            return (
                True,
                "numeric_loose",
                0.95,
            )

    overlap = overlap_score(
        predicted,
        gold,
    )

    if overlap >= 0.50:

        return (
            True,
            "narrative",
            overlap,
        )

    if overlap >= 0.30:

        return (
            True,
            "partial",
            overlap,
        )

    return (
        False,
        "wrong",
        overlap,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Load Questions
# ─────────────────────────────────────────────────────────────────────────────


def load_questions(
    limit: int = 0,
):

    validate_financebench()

    questions = []

    with open(
        FB_QUESTIONS,
        encoding="utf-8",
    ) as f:

        for line in f:

            questions.append(
                json.loads(line)
            )

    metadata = []

    with open(
        FB_METADATA,
        encoding="utf-8",
    ) as f:

        for line in f:

            metadata.append(
                json.loads(line)
            )

    meta_by_doc = {
        m["doc_name"]: m
        for m in metadata
    }

    for q in questions:

        meta = meta_by_doc.get(
            q.get("doc_name", ""),
            {},
        )

        q["company_name"] = meta.get(
            "company",
            "UNKNOWN",
        )

        q["doc_type"] = meta.get(
            "doc_type",
            "10-K",
        )

        q["fiscal_year"] = (
            f"FY{meta.get('doc_period', '')}"
        )

        q["pdf_path"] = str(
            FB_PDFS_DIR
            / f"{q['doc_name']}.pdf"
        )

        q["gold_answer"] = str(
            q.get("answer", "")
        )

    if limit > 0:

        questions = questions[
            :limit
        ]

    return questions

# ─────────────────────────────────────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────────────────────────────────────


def run_eval(args):

    # ── LLM PREFLIGHT (2026-06-21) ──────────────────────────────────────
    # The #1 cause of fake-low scores (13.3%) was Ollama being down or the
    # model name not matching what was pulled -> is_available() False ->
    # every question silently became RETRIEVAL_MISS. Check LOUDLY up front
    # and prewarm, so a broken LLM aborts in 10s instead of wasting 27 min.
    if not os.environ.get("SNIPER_ONLY"):
        try:
            from src.utils.llm_client import get_llm_client, prewarm_model, DEFAULT_MODEL
            client = get_llm_client()
            print(f"[PREFLIGHT] Checking Ollama for model '{DEFAULT_MODEL}' ...")
            if not client.is_available():
                print("=" * 80)
                print("  LLM PREFLIGHT FAILED - Ollama not available or model missing")
                print(f"  Expected model: {DEFAULT_MODEL}")
                print("  Fix: start `ollama serve` and run `ollama pull "
                      f"{DEFAULT_MODEL}`")
                print("  (or run with --sniper-only to skip the LLM entirely)")
                print("=" * 80)
                raise SystemExit("LLM preflight failed - aborting to avoid a fake-low score.")
            print("[PREFLIGHT] Ollama OK. Prewarming model (cold-start can take 60-120s)...")
            if prewarm_model():
                print("[PREFLIGHT] Model warm. Proceeding with eval.")
            else:
                print("[PREFLIGHT] WARNING: prewarm returned no output, continuing anyway.")
        except SystemExit:
            raise
        except Exception as _e:
            print(f"[PREFLIGHT] WARNING: could not run LLM preflight ({_e}). Continuing.")

    questions = load_questions(
        limit=args.limit,
    )

    by_doc = defaultdict(list)

    for q in questions:

        by_doc[
            q["doc_name"]
        ].append(q)

    logger.info(
        "Documents: %d",
        len(by_doc),
    )

    results = []

    output_path = (
        RESULTS_DIR
        / (
            "financebench_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    )

    partial_path = str(
        output_path
    ).replace(
        ".json",
        "_partial.json",
    )

    total_start = time.time()

    for idx, (
        doc_name,
        q_list,
    ) in enumerate(
        by_doc.items(),
        start=1,
    ):

        company = q_list[0][
            "company_name"
        ]

        fiscal_year = q_list[0][
            "fiscal_year"
        ]

        pdf_path = q_list[0][
            "pdf_path"
        ]

        print()
        print("=" * 80)
        print(
            f"[{idx}/{len(by_doc)}] "
            f"{company} | "
            f"{doc_name} | "
            f"{len(q_list)} questions"
        )
        print("=" * 80)

        if not os.path.exists(
            pdf_path
        ):

            logger.warning(
                "Missing PDF: %s",
                pdf_path,
            )

            continue

        try:

            ingest_start = time.time()

            pipeline = FinBenchPipeline()

            state = pipeline.ingest(
                document_path=pdf_path,
                session_id=f"fb-{idx}",
                company_name=company,
                doc_type="10-K",
                fiscal_year=fiscal_year,
            )

            ingest_elapsed = (
                time.time()
                - ingest_start
            )

            logger.info(
                "Ingest %.1fs | chunks=%d",
                ingest_elapsed,
                getattr(
                    state,
                    "chunk_count",
                    0,
                ),
            )

        except Exception as exc:

            logger.exception(
                "Ingest failed"
            )

            print(
                f"INGEST FAILED: {exc}"
            )

            continue

        # ─────────────────────────────────────────────────────────────────
        # Questions
        # ─────────────────────────────────────────────────────────────────

        for q in q_list:

            q_start = time.time()

            try:

                result_state = (
                    pipeline.query(
                        state,
                        q["question"],
                    )
                )

                elapsed = (
                    time.time()
                    - q_start
                )

                pred = str(
                    getattr(
                        result_state,
                        "final_answer",
                        "",
                    )
                    or ""
                )

                correct, match_type, score = (
                    is_correct(
                        pred,
                        q["gold_answer"],
                    )
                )

                pod = getattr(
                    result_state,
                    "winning_pod",
                    "?",
                )

                results.append(
                    {
                        "financebench_id": q[
                            "financebench_id"
                        ],
                        "question": q[
                            "question"
                        ][:300],
                        "gold": q[
                            "gold_answer"
                        ][:300],
                        "predicted": pred[
                            :1000
                        ],
                        "correct": correct,
                        "match_type": match_type,
                        "score": round(
                            score,
                            3,
                        ),
                        "elapsed_sec": round(
                            elapsed,
                            2,
                        ),
                        "winning_pod": pod,
                    }
                )

                mark = (
                    "✓"
                    if correct
                    else "✗"
                )

                print(
                    f"{mark} "
                    f"{elapsed:>6.1f}s "
                    f"{match_type:<18} "
                    f"{pod:<20} "
                    f"{pred[:80]}"
                )

            except Exception as exc:

                logger.exception(
                    "Question failed"
                )

                results.append(
                    {
                        "financebench_id": q[
                            "financebench_id"
                        ],
                        "question": q[
                            "question"
                        ][:300],
                        "gold": q[
                            "gold_answer"
                        ][:300],
                        "predicted": "",
                        "correct": False,
                        "match_type": "exception",
                        "error": str(exc),
                    }
                )

            # Autosave

            with open(
                partial_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    results,
                    f,
                    indent=2,
                )

        # Cleanup between docs

        gc.collect()

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        except Exception:
            pass

    total_elapsed = (
        time.time()
        - total_start
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    if os.path.exists(
        partial_path
    ):

        os.remove(
            partial_path
        )

    print()
    print("=" * 80)
    print(
        f"COMPLETE | "
        f"{len(results)} questions | "
        f"{total_elapsed / 60:.1f} min"
    )
    print(output_path)
    print("=" * 80)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────


def _classify_failure(r):
    """GOLD/SILVER/DIAMOND triage (babe's idea, 2026-06-13).

    Every run auto-sorts each MISS into a fix-priority bucket so we always
    know WHAT to fix next instead of guessing:
      GOLD    = correct number present but format/grader mismatch → easy, high value
      SILVER  = right approach, wrong cell/period/value         → medium
      DIAMOND = genuine reasoning miss / abstain / LLM ceiling   → hard
    """
    pred = str(r.get("predicted", "") or "")
    gold = str(r.get("gold", "") or "")
    mt   = r.get("match_type", "")

    if r.get("correct"):
        return None

    # DIAMOND: nothing useful produced (abstain / retrieval miss / empty)
    if mt in ("non_answer", "empty", "exception") or _is_non_answer(pred) \
            or "RETRIEVAL_MISS" in pred:
        return "DIAMOND"

    pred_num = normalize_number(pred)
    gold_num = normalize_number(gold)

    # GOLD: both have numbers and they're CLOSE but the grader didn't pass
    # (scale / format / rounding) → the value is essentially right.
    if pred_num and gold_num:
        try:
            pf, gf = float(pred_num), float(gold_num)
            if abs(gf) > 1e-9 and abs(pf - gf) / abs(gf) < 0.10:
                return "GOLD"          # within 10% — format/scale issue
        except Exception:
            pass
        return "SILVER"                # both numeric but far apart → wrong cell

    # SILVER: we produced a numeric answer but gold is text (or vice-versa)
    if pred_num or gold_num:
        return "SILVER"

    # DIAMOND: pure text-vs-text reasoning miss
    return "DIAMOND"


def write_summary(
    results,
    output_dir,
):

    n = len(results)

    if n == 0:
        return

    correct = sum(
        1
        for r in results
        if r.get("correct")
    )

    accuracy = correct / n

    # babe's GOLD/SILVER/DIAMOND triage
    buckets = {"GOLD": [], "SILVER": [], "DIAMOND": []}
    for r in results:
        tier = _classify_failure(r)
        if tier:
            buckets[tier].append(r)

    summary_path = (
        output_dir
        / (
            "financebench_summary_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
    )

    lines = [
        "# FinanceBench Results",
        "",
        f"Questions: {n}",
        f"Correct: {correct}",
        f"Accuracy: {100 * accuracy:.2f}%",
        "",
        "## Fix-Priority Triage (GOLD = easiest wins)",
        "",
        f"🥇 GOLD   ({len(buckets['GOLD'])}): right number, format/scale mismatch — fix the grader/formatter",
        f"🥈 SILVER ({len(buckets['SILVER'])}): right approach, wrong cell/period — fix extraction targeting",
        f"💎 DIAMOND ({len(buckets['DIAMOND'])}): reasoning miss / abstain — LLM ceiling, hardest",
        "",
        "### 🥇 GOLD fixes (do these first — points are basically earned):",
    ]
    for r in buckets["GOLD"]:
        lines.append(f"- PRED `{str(r.get('predicted',''))[:45]}` | GOLD `{str(r.get('gold',''))[:45]}`")
    lines.append("")
    lines.append("### 🥈 SILVER fixes (wrong cell/period):")
    for r in buckets["SILVER"]:
        lines.append(f"- PRED `{str(r.get('predicted',''))[:45]}` | GOLD `{str(r.get('gold',''))[:45]}`")
    lines.append("")
    lines.append("### 💎 DIAMOND (LLM-ceiling reasoning):")
    for r in buckets["DIAMOND"]:
        lines.append(f"- PRED `{str(r.get('predicted',''))[:45]}` | GOLD `{str(r.get('gold',''))[:45]}`")

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(f"🥇 GOLD (easy):   {len(buckets['GOLD'])}")
    print(f"🥈 SILVER (med):  {len(buckets['SILVER'])}")
    print(f"💎 DIAMOND (hard): {len(buckets['DIAMOND'])}")
    print("=" * 80)
    print(summary_path)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--sniper-only",
        action="store_true",
        help="Skip all LLM pods; Sniper-only fast path",
    )

    args = parser.parse_args()

    if args.sniper_only:
        os.environ["SNIPER_ONLY"] = "1"
        os.environ["DISABLE_BGE"] = "1"
        print("[P0] SNIPER-ONLY mode: LLM + BGE disabled")

    print("=" * 80)
    print(
        "🚀 FULL FINANCEBENCH EVAL"
    )
    print("=" * 80)

    print(
        f"Seed={args.seed} | "
        f"Limit={args.limit or 'ALL'}"
    )

    results = run_eval(args)

    write_summary(
        results,
        RESULTS_DIR,
    )


if __name__ == "__main__":
    main()