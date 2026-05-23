"""
eval/run_financebench.py

Production-safe FinanceBench evaluation runner.

Features
--------
- Auto-detects FinanceBench dataset
- Auto-clones dataset if missing
- Works in Colab / Linux / Windows
- Better diagnostics
- Partial result auto-save
- Robust failure handling
- Optional Ollama health check
- Sniper-only fast mode

Usage
-----
python eval/run_financebench.py
python eval/run_financebench.py --limit 10
python eval/run_financebench.py --skip-llm-check
python eval/run_financebench.py --sniper-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time

from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import FinBenchPipeline

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("financebench_eval")
logger.setLevel(logging.INFO)

# =============================================================================
# Paths
# =============================================================================

RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINANCEBENCH_REPO = "https://github.com/patronus-ai/financebench.git"

DATASET_CANDIDATES = [
    ROOT / "financebench_dataset" / "financebench" / "data" / "financebench_open_source.jsonl",
    ROOT / "financebench" / "data" / "financebench_open_source.jsonl",
    Path("/content/finbench_agent/financebench_dataset/financebench/data/financebench_open_source.jsonl"),
    Path("/content/financebench_dataset/financebench/data/financebench_open_source.jsonl"),
]

# =============================================================================
# Dataset resolver
# =============================================================================


def ensure_financebench_dataset() -> Path:
    """
    Locate or auto-download FinanceBench dataset.
    """

    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            print(f"✅ Dataset found:\n   {candidate}")
            return candidate

    clone_dir = ROOT / "financebench_dataset"

    print("\n📦 FinanceBench dataset not found")
    print(f"📥 Cloning into:\n   {clone_dir}\n")

    try:
        subprocess.run(
            [
                "git",
                "clone",
                FINANCEBENCH_REPO,
                str(clone_dir),
            ],
            check=True,
        )

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "\n❌ Failed to clone FinanceBench dataset.\n"
            "Check:\n"
            "  • internet connection\n"
            "  • git installation\n"
            "  • GitHub access\n"
        ) from exc

    dataset_path = (
        clone_dir
        / "financebench"
        / "data"
        / "financebench_open_source.jsonl"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            "\n❌ Dataset cloned but JSONL missing.\n"
            f"Expected:\n   {dataset_path}\n"
        )

    print(f"✅ Dataset ready:\n   {dataset_path}")

    return dataset_path


DATASET_PATH = ensure_financebench_dataset()

# =============================================================================
# Ollama health check
# =============================================================================


def check_ollama_running() -> tuple[bool, str]:
    """
    Check if Ollama is alive and model exists.
    """

    try:
        import urllib.request

        with urllib.request.urlopen(
            "http://localhost:11434/api/tags",
            timeout=5,
        ) as resp:

            if resp.status != 200:
                return False, f"Ollama HTTP {resp.status}"

            data = json.loads(resp.read().decode())

            models = [
                m.get("name", "")
                for m in data.get("models", [])
            ]

            llama_models = [
                m for m in models
                if "llama3.1:8b" in m.lower()
            ]

            if not llama_models:
                return (
                    False,
                    (
                        "Ollama running but llama3.1:8b missing.\n"
                        f"Available models: {models}\n"
                        "Run:\n"
                        "  ollama pull llama3.1:8b"
                    ),
                )

            model_name = llama_models[0]

            print(f"⏳ Warming up model: {model_name}")

            payload = json.dumps(
                {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": "hello",
                        }
                    ],
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "num_predict": 5,
                    },
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120):
                pass

            return True, f"Ollama OK | model={model_name}"

    except Exception as exc:
        return False, str(exc)


# =============================================================================
# Helpers
# =============================================================================


def normalize_num(s: str) -> str:
    if not s:
        return ""

    s = (
        str(s)
        .replace(",", "")
        .replace("$", "")
        .replace("%", "")
    )

    s = re.sub(
        r"\s+(million|billion|thousand|m|bn|mn)\b",
        "",
        s,
        flags=re.I,
    )

    nums = re.findall(r"-?\d+\.?\d*", s)

    if not nums:
        return ""

    try:
        return (
            f"{float(nums[0]):.4f}"
            .rstrip("0")
            .rstrip(".")
        )
    except Exception:
        return ""


def numbers_match(
    predicted: str,
    gold: str,
    tol: float = 0.02,
) -> bool:

    p = normalize_num(predicted)
    g = normalize_num(gold)

    if not p or not g:
        return False

    try:
        pf = float(p)
        gf = float(g)

        if abs(gf) < 1e-9:
            return abs(pf) < 1e-3

        return abs(pf - gf) / abs(gf) < tol

    except Exception:
        return False


def is_correct(
    predicted: str,
    gold: str,
) -> tuple[bool, str]:

    if not predicted or not predicted.strip():
        return False, "empty"

    if "INSUFFICIENT_DATA" in predicted:
        return False, "refused"

    if "LLM_UNAVAILABLE" in predicted:
        return False, "llm_dead"

    if "RETRIEVAL_MISS" in predicted:
        return False, "retrieval_miss"

    if numbers_match(predicted, gold, 0.02):
        return True, "numeric"

    if numbers_match(predicted, gold, 0.05):
        return True, "numeric_loose"

    return False, "wrong"


# =============================================================================
# Dataset loader
# =============================================================================


def load_questions(limit: int = 0) -> list[dict]:
    """
    Load FinanceBench JSONL.
    """

    print(f"Dataset: {DATASET_PATH}")

    questions = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                questions.append(json.loads(line))

            except json.JSONDecodeError as exc:
                logger.warning(
                    "Bad JSON line %s: %s",
                    idx,
                    exc,
                )

    if limit > 0:
        questions = questions[:limit]

    print(f"✅ Loaded {len(questions)} questions")

    return questions


# =============================================================================
# Main eval
# =============================================================================


def run_eval(args) -> list[dict]:

    if not args.skip_llm_check:

        print("\n" + "=" * 78)
        print("🔌 Checking Ollama")
        print("=" * 78)

        ok, msg = check_ollama_running()

        if ok:
            print(f"✅ {msg}")
        else:
            print(f"⚠️ {msg}")

    if not os.environ.get("BGE_DEVICE"):
        os.environ["DISABLE_BGE"] = "1"
        print("⚡ BGE disabled for CPU eval")

    questions = load_questions(args.limit)

    print(f"📊 Questions: {len(questions)}")

    results = []

    output_path = (
        RESULTS_DIR
        / f"financebench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    pipeline = FinBenchPipeline()

    total_start = time.time()

    for idx, q in enumerate(questions, start=1):

        print("\n" + "-" * 80)

        qid = q.get("qid", f"q{idx}")

        question = q.get("question", "")
        gold = q.get("answer", "")

        print(f"[{idx}/{len(questions)}] {qid}")
        print(question[:120])

        t0 = time.time()

        try:

            state = pipeline.query(question)

            pred = str(
                getattr(state, "final_answer", "")
                or ""
            )

            correct, match_type = is_correct(
                pred,
                gold,
            )

            elapsed = time.time() - t0

            result = {
                "qid": qid,
                "question": question,
                "gold": gold,
                "predicted": pred,
                "correct": correct,
                "match_type": match_type,
                "elapsed_sec": round(elapsed, 1),
            }

            results.append(result)

            mark = "✓" if correct else "✗"

            print(
                f"{mark} "
                f"{match_type:<14} "
                f"{elapsed:>5.1f}s "
                f"pred={pred[:80]}"
            )

        except Exception as exc:

            print(f"✗ ERROR: {exc}")

            results.append(
                {
                    "qid": qid,
                    "question": question,
                    "gold": gold,
                    "predicted": "",
                    "correct": False,
                    "match_type": "exception",
                    "error": str(exc),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("✅ EVAL COMPLETE")
    print("=" * 80)

    print(f"Questions: {len(results)}")

    correct = sum(
        1 for r in results
        if r.get("correct")
    )

    accuracy = (
        100 * correct / len(results)
        if results else 0
    )

    print(f"Correct:  {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Time:     {total_time / 60:.1f} min")

    print(f"\nSaved:\n{output_path}")

    return results


# =============================================================================
# Summary
# =============================================================================


def write_summary(results: list[dict]) -> Path:

    summary_path = (
        RESULTS_DIR
        / f"financebench_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    total = len(results)

    correct = sum(
        1 for r in results
        if r.get("correct")
    )

    accuracy = (
        100 * correct / total
        if total else 0
    )

    lines = []

    lines.append("# FinanceBench Eval Summary")
    lines.append("")
    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")
    lines.append(f"- Questions: {total}")
    lines.append(f"- Correct:   {correct}")
    lines.append(f"- Accuracy:  {accuracy:.1f}%")
    lines.append("")

    by_match = defaultdict(int)

    for r in results:
        by_match[r.get("match_type", "?")] += 1

    lines.append("## Match Types")
    lines.append("")

    for k, v in sorted(
        by_match.items(),
        key=lambda x: -x[1],
    ):
        lines.append(f"- {k}: {v}")

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"\n📝 Summary saved:\n{summary_path}")

    return summary_path


# =============================================================================
# CLI
# =============================================================================


def main():

    parser = argparse.ArgumentParser(
        description="FinanceBench Evaluation Runner"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit questions",
    )

    parser.add_argument(
        "--skip-llm-check",
        action="store_true",
        help="Skip Ollama health check",
    )

    parser.add_argument(
        "--sniper-only",
        action="store_true",
        help="Use sniper-only retrieval",
    )

    args = parser.parse_args()

    print("=" * 78)
    print("🚀 FULL FINANCEBENCH EVAL")
    print("=" * 78)

    results = run_eval(args)

    write_summary(results)


if __name__ == "__main__":
    main()