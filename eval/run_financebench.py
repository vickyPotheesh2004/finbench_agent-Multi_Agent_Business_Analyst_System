from __future__ import annotations

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["CHROMADB_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
import argparse
import gc
import json
import logging
import os
import re
import sys
import time

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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

from src.utils.runtime_cache import (
    RuntimeCache,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
)

logger = logging.getLogger(
    "financebench_eval"
)

logger.setLevel(logging.INFO)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset Paths
# ──────────────────────────────────────────────────────────────────────────────

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
    FB_ROOT / "pdfs"
)

# ──────────────────────────────────────────────────────────────────────────────
# Results Directory
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────


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

        print("\n" + "=" * 80)
        print("❌ FINANCEBENCH DATASET MISSING")
        print("=" * 80)

        print()

        for item in missing:
            print(f" - {item}")

        print()

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

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def normalize_number(text: str):

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

    p = normalize_number(predicted)
    g = normalize_number(gold)

    if not p or not g:
        return False

    try:

        pf = float(p)
        gf = float(g)

        if abs(gf) < 1e-9:
            return abs(pf) < 1e-3

        return (
            abs(pf - gf) / abs(gf)
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


def is_correct(
    predicted: str,
    gold: str,
):

    if not predicted:
        return False, "empty", 0.0

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
            return True, "numeric", 1.0

        if numbers_match(
            pred_num,
            gold_num,
            tolerance=0.05,
        ):
            return True, "numeric_loose", 0.95

    overlap = overlap_score(
        predicted,
        gold,
    )

    if overlap >= 0.50:
        return True, "narrative", overlap

    if overlap >= 0.30:
        return True, "partial", overlap

    return False, "wrong", overlap

# ──────────────────────────────────────────────────────────────────────────────
# Load Questions
# ──────────────────────────────────────────────────────────────────────────────


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
        questions = questions[:limit]

    return questions

# ──────────────────────────────────────────────────────────────────────────────
# Eval
# ──────────────────────────────────────────────────────────────────────────────


def run_eval(args):

    questions = load_questions(
        limit=args.limit,
    )

    by_doc = defaultdict(list)

    for q in questions:
        by_doc[q["doc_name"]].append(q)

    logger.info(
        "Documents: %d",
        len(by_doc),
    )

    results = []

    pipeline_cache = {}

    output_path = (
        RESULTS_DIR
        / f"financebench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    partial_path = str(output_path).replace(
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

            if pdf_path in pipeline_cache:

                pipeline, state = (
                    pipeline_cache[pdf_path]
                )

                logger.info(
                    "Using cached ingest"
                )

            else:

                pipeline = FinBenchPipeline()

                ingest_start = time.time()

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

                pipeline_cache[pdf_path] = (
                    pipeline,
                    state,
                )

        except Exception as exc:

            logger.exception(
                "Ingest failed"
            )

            print(
                f"INGEST FAILED: {exc}"
            )

            continue

        # ──────────────────────────────────────────────────────────────────────
        # Questions
        # ──────────────────────────────────────────────────────────────────────

        for q in q_list:

            q_start = time.time()

            try:

                result_state = pipeline.query(
                    state,
                    q["question"],
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
                        ][:200],
                        "gold": q[
                            "gold_answer"
                        ][:200],
                        "predicted": pred[:400],
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
                    f"{elapsed:>5.1f}s "
                    f"{match_type:<18} "
                    f"{pod:<18} "
                    f"{pred[:70]}"
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
                        ][:200],
                        "gold": q[
                            "gold_answer"
                        ][:200],
                        "predicted": "",
                        "correct": False,
                        "match_type": "exception",
                        "error": str(exc),
                    }
                )

            # autosave

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

        gc.collect()

    total_elapsed = (
        time.time() - total_start
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
        os.remove(partial_path)

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

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────


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

    summary_path = (
        output_dir
        / f"financebench_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    lines = [
        "# FinanceBench Results",
        "",
        f"Questions: {n}",
        f"Correct: {correct}",
        f"Accuracy: {100 * accuracy:.2f}%",
    ]

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(summary_path)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


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

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 FULL FINANCEBENCH EVAL")
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