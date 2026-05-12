"""
eval/run_financebench.py — Official FinanceBench Evaluation

Runs FinBench pipeline against Patronus AI's FinanceBench 150-question
test set. Compares to 2026 frontier baselines.

Usage:
    python eval/run_financebench.py --seed 42
    python eval/run_financebench.py --limit 5
    python eval/run_financebench.py --resume

Baselines (April 2026 awesomeagents.ai/leaderboards):
    o3:              ~90%      GPT-5:           ~88%
    GPT-4.1:         ~85%      Claude 4 Opus:   ~82%
    Gemini 2.5 Pro:  ~80%      Claude 4 Sonnet: ~78%
    BloombergGPT:    ~53%      FinGPT:          ~40%
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import FinBenchPipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("financebench_eval")
logger.setLevel(logging.INFO)

FB_ROOT = ROOT / "financebench_dataset" / "financebench"
FB_QUESTIONS = FB_ROOT / "data" / "financebench_open_source.jsonl"
FB_METADATA  = FB_ROOT / "data" / "financebench_document_information.jsonl"
FB_PDFS_DIR  = FB_ROOT / "pdfs"

# Auto-detect: if Google Drive is mounted (Colab), save there for persistence
# Otherwise save to local eval/results
import os as _os
if _os.path.exists("/content/drive/MyDrive"):
    RESULTS_DIR = Path("/content/drive/MyDrive/finbench_results")
else:
    RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_number(text: str) -> str:
    if not text: return ""
    s = str(text).replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\s+(million|billion|thousand|m|bn|mn)\b", "", s, flags=re.I)
    nums = re.findall(r"-?\d+\.?\d*", s)
    if nums:
        try:
            v = float(nums[0])
            return f"{v:.4f}".rstrip("0").rstrip(".") or "0"
        except ValueError:
            return ""
    return ""


def extract_first_number(text: str) -> str:
    if not text: return ""
    nums = re.findall(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+", text)
    return nums[0] if nums else ""


def numbers_match(predicted: str, gold: str, tolerance: float = 0.02) -> bool:
    p, g = normalize_number(predicted), normalize_number(gold)
    if not p or not g: return False
    try:
        pf, gf = float(p), float(g)
        if abs(gf) < 1e-9: return abs(pf) < 1e-3
        return abs(pf - gf) / abs(gf) < tolerance
    except (ValueError, TypeError):
        return False


def text_overlap_score(predicted: str, gold: str) -> float:
    if not predicted or not gold: return 0.0
    p = set(re.findall(r"\w+", predicted.lower()))
    g = set(re.findall(r"\w+", gold.lower()))
    if not g: return 0.0
    return len(p & g) / len(p | g) if (p | g) else 0.0


def is_correct(predicted: str, gold: str) -> tuple:
    if not predicted or not predicted.strip():
        return False, "empty", 0.0
    if "INSUFFICIENT_DATA" in predicted or "LLM_UNAVAILABLE" in predicted:
        return False, "refused", 0.0
    pred_num = extract_first_number(predicted)
    gold_num = extract_first_number(gold)
    if gold_num and pred_num:
        if numbers_match(pred_num, gold_num):
            return True, "numeric", 1.0
        if numbers_match(pred_num, gold_num, tolerance=0.05):
            return True, "numeric_loose", 0.95
    overlap = text_overlap_score(predicted, gold)
    if overlap >= 0.50: return True, "narrative", overlap
    if not gold_num and overlap >= 0.30: return True, "narrative_partial", overlap
    return False, "wrong", overlap


def load_questions(limit: int = 0) -> list:
    if not FB_QUESTIONS.exists():
        raise FileNotFoundError(
            f"FinanceBench not found at {FB_QUESTIONS}. Clone the dataset first."
        )
    questions = [json.loads(line) for line in open(FB_QUESTIONS, encoding="utf-8")]
    meta_records = [json.loads(line) for line in open(FB_METADATA, encoding="utf-8")]
    meta_by_doc  = {m["doc_name"]: m for m in meta_records}
    for q in questions:
        m = meta_by_doc.get(q.get("doc_name", ""), {})
        q["company_name"] = m.get("company", "UNKNOWN")
        q["doc_type"]     = m.get("doc_type", "10-K")
        q["fiscal_year"]  = f"FY{m.get('doc_period', 0)}"
        q["pdf_path"]     = str(FB_PDFS_DIR / f"{q.get('doc_name', '')}.pdf")
        q["gold_answer"]  = str(q.get("answer", ""))
    if limit > 0:
        questions = questions[:limit]
    return questions


def _short_id(fid) -> str:
    """Normalize financebench_id (which is a string) to a short display label."""
    s = str(fid)
    # Strip "financebench_id_" prefix if present
    if s.startswith("financebench_id_"):
        s = s[len("financebench_id_"):]
    return s[:10].ljust(10)


def run_eval(args) -> list:
    questions = load_questions(limit=args.limit)
    logger.info("Loaded %d FinanceBench questions", len(questions))
    by_doc = defaultdict(list)
    for q in questions:
        by_doc[q["doc_name"]].append(q)
    logger.info("Unique documents: %d", len(by_doc))

    results = []
    seen_ids = set()
    if args.resume:
        partial_files = sorted(RESULTS_DIR.glob("financebench_*_partial.json"))
        if partial_files:
            with open(partial_files[-1]) as f:
                results = json.load(f)
            seen_ids = {r["financebench_id"] for r in results}
            logger.info("Resumed with %d done", len(results))

    t_start = time.time()
    output_path = RESULTS_DIR / f"financebench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    partial_path = str(output_path).replace(".json", "_partial.json")

    for doc_idx, (doc_name, q_list) in enumerate(by_doc.items(), 1):
        if all(q["financebench_id"] in seen_ids for q in q_list):
            continue
        pdf_path = q_list[0]["pdf_path"]
        if not os.path.exists(pdf_path):
            logger.warning("Missing PDF: %s", pdf_path)
            for q in q_list:
                results.append({
                    "financebench_id": q["financebench_id"],
                    "question": q["question"][:200], "gold": q["gold_answer"][:200],
                    "predicted": "", "error": "missing_pdf",
                    "correct": False, "match_type": "missing_pdf",
                })
            continue

        company = q_list[0]["company_name"]
        fy = q_list[0]["fiscal_year"]
        print()
        print("=" * 80)
        print(f" [{doc_idx}/{len(by_doc)}] {company} — {doc_name} ({len(q_list)} Q)")
        print("=" * 80)

        t0 = time.time()
        try:
            pipeline = FinBenchPipeline()
            state = pipeline.ingest(
                document_path=pdf_path, session_id=f"fb-{doc_idx}",
                company_name=company, doc_type="10-K", fiscal_year=fy,
            )
            print(f"   Ingest: {time.time()-t0:.1f}s | chunks={state.chunk_count} | cells={len(state.table_cells)}")
        except Exception as exc:
            print(f"   Ingest FAILED: {exc}")
            for q in q_list:
                if q["financebench_id"] in seen_ids: continue
                results.append({
                    "financebench_id": q["financebench_id"],
                    "question": q["question"][:200], "gold": q["gold_answer"][:200],
                    "predicted": "", "error": f"ingest: {exc}",
                    "correct": False, "match_type": "ingest_error",
                })
            continue

        for q in q_list:
            if q["financebench_id"] in seen_ids: continue
            t0 = time.time()
            try:
                state = pipeline.query(state, q["question"])
                elapsed = time.time() - t0
                pred = str(getattr(state, "final_answer", "") or "")
                correct, match_type, score = is_correct(pred, q["gold_answer"])
                pod = getattr(state, "winning_pod", "?") or "?"
                results.append({
                    "financebench_id": q["financebench_id"],
                    "question": q["question"][:200], "gold": q["gold_answer"][:200],
                    "predicted": pred[:400], "winning_pod": pod,
                    "correct": correct, "match_type": match_type,
                    "score": round(score, 3), "elapsed_sec": round(elapsed, 1),
                })
                mark = "✓" if correct else "✗"
                # Fix Bug 1: financebench_id is a STRING, not int — use _short_id helper
                fid = _short_id(q["financebench_id"])
                print(f"   {mark} {fid} ({match_type:>16}) {elapsed:>5.1f}s  {pod[:18]:<18}  pred={pred[:60]}")
            except Exception as exc:
                fid = _short_id(q.get("financebench_id", "?"))
                print(f"   ✗ {fid} ERROR: {exc}")
                results.append({
                    "financebench_id": q.get("financebench_id", "?"),
                    "question": q.get("question", "")[:200],
                    "gold": q.get("gold_answer", "")[:200],
                    "error": str(exc)[:300], "correct": False, "match_type": "exception",
                })

        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    if os.path.exists(partial_path):
        os.remove(partial_path)

    print()
    print("=" * 80)
    print(f"  EVAL COMPLETE: {len(results)} questions in {total_time/60:.1f} min")
    print(f"  Saved: {output_path}")
    print("=" * 80)
    return results


PUBLISHED_BASELINES = [
    ("o3 (OpenAI, reasoning)",         0.90,  "API",          "Cloud"),
    ("GPT-5 (OpenAI)",                 0.88,  "API",          "Cloud"),
    ("GPT-4.1 (OpenAI)",               0.85,  "API",          "Cloud"),
    ("Claude 4 Opus (Anthropic)",      0.82,  "API",          "Cloud"),
    ("Gemini 2.5 Pro (Google)",        0.80,  "API",          "Cloud"),
    ("Claude 4 Sonnet (Anthropic)",    0.78,  "API",          "Cloud"),
    ("BloombergGPT (50B fine-tune)",   0.53,  "Proprietary",  "Cloud"),
    ("FinGPT (open-source fine-tune)", 0.40,  "Open weights", "GPU req"),
]


def write_summary(results: list, output_dir: Path) -> Path:
    n = len(results)
    if n == 0: return None
    correct = sum(1 for r in results if r.get("correct"))
    errors = sum(1 for r in results if "error" in r and not r.get("correct"))
    refused = sum(1 for r in results if r.get("match_type") == "refused")
    numeric_hit = sum(1 for r in results if r.get("match_type") in ("numeric", "numeric_loose"))
    narrative_hit = sum(1 for r in results if r.get("match_type") in ("narrative", "narrative_partial"))
    accuracy = correct / n

    summary_path = output_dir / f"financebench_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines = []
    lines.append("# FinBench on Official FinanceBench")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Questions evaluated:** {n} of 150")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"- ✅ **Correct:**             {correct}/{n}  ({100*accuracy:.1f}%)")
    lines.append(f"- ✗ Wrong/refused/error:    {n - correct}/{n}")
    lines.append(f"  - Numeric hits:           {numeric_hit}")
    lines.append(f"  - Narrative hits:         {narrative_hit}")
    lines.append(f"  - Refused (no data):      {refused}")
    lines.append(f"  - Errors:                 {errors}")
    lines.append("")
    times = [r.get("elapsed_sec", 0) for r in results if "elapsed_sec" in r]
    if times:
        lines.append(f"- Avg time/Q:  {sum(times)/len(times):.1f}s")
        lines.append(f"- Total time:  {sum(times)/60:.1f} min")
        lines.append("")

    lines.append("## Comparison To Published Baselines (April 2026)")
    lines.append("")
    lines.append("| System | FinanceBench | Type | Location |")
    lines.append("|--------|--------------|------|----------|")
    lines.append(f"| **FinBench (this work)** | **{100*accuracy:.1f}%** | **Open-source** | **Laptop** |")
    for name, score, kind, loc in PUBLISHED_BASELINES:
        lines.append(f"| {name} | {100*score:.0f}% | {kind} | {loc} |")
    lines.append("")

    lines.append("## Failure Mode Distribution")
    failure_modes = defaultdict(int)
    for r in results:
        if not r.get("correct"):
            failure_modes[r.get("match_type", "unknown")] += 1
    for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
        lines.append(f"- {mode}: {count}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary: {summary_path}")
    print("\n".join(lines))
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Run FinBench on FinanceBench")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--limit",  type=int, default=0, help="0 = all 150")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(f"FinanceBench evaluation | seed={args.seed} | limit={args.limit or 'all 150'}")
    print(f"Dataset: {FB_QUESTIONS}")
    results = run_eval(args)
    write_summary(results, RESULTS_DIR)


if __name__ == "__main__":
    main()