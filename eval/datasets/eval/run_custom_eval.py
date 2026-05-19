"""
eval/run_custom_eval.py — Internal Benchmark Eval Runner

Runs FinBench pipeline against your hand-crafted JSONL datasets in 
eval/datasets/. Each JSONL file represents one company's 10-K filing.

Usage:
    python eval/run_custom_eval.py                  # all datasets
    python eval/run_custom_eval.py --company AAPL    # one company
    python eval/run_custom_eval.py --limit 5         # smoke test

Output:
    eval/results/custom_<timestamp>.json
    eval/results/custom_summary_<timestamp>.md
"""

from __future__ import annotations
import argparse, json, logging, os, re, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import FinBenchPipeline

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("custom_eval")
logger.setLevel(logging.INFO)

DATASETS_DIR = ROOT / "eval" / "datasets"
DOCS_DIR = ROOT / "documents" / "sec_filings"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Company → document file mapping
COMPANY_DOCS = {
    "apple":     "AAPL_FY2023_10-K.html",
    "microsoft": "MSFT_FY2023_10-K.html",
    "nvidia":    "NVDA_FY2024_10-K.html",
    "amazon":    "AMZN_FY2023_10-K.html",
    "alphabet":  "GOOG_FY2023_10-K.html",
    "meta":      "META_FY2023_10-K.html",
    "tesla":     "TSLA_FY2023_10-K.html",
}


def normalize_num(s: str) -> str:
    """Strip commas, $, %, common units."""
    if not s: return ""
    s = str(s).replace(",", "").replace("$", "").replace("%", "")
    s = re.sub(r"\s+(million|billion|thousand|m|bn|mn)\b", "", s, flags=re.I)
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums: return ""
    try:
        return f"{float(nums[0]):.4f}".rstrip("0").rstrip(".") or "0"
    except ValueError:
        return ""


def numbers_match(predicted: str, gold: str, tol: float = 0.02) -> bool:
    """2% tolerance for rounding."""
    p, g = normalize_num(predicted), normalize_num(gold)
    if not p or not g: return False
    try:
        pf, gf = float(p), float(g)
        if abs(gf) < 1e-9: return abs(pf) < 1e-3
        return abs(pf - gf) / abs(gf) < tol
    except (ValueError, TypeError):
        return False


def is_correct(predicted: str, gold: str) -> tuple:
    if not predicted or not predicted.strip():
        return False, "empty"
    if "INSUFFICIENT_DATA" in predicted or "LLM_UNAVAILABLE" in predicted:
        return False, "refused"
    if numbers_match(predicted, gold):
        return True, "numeric"
    if numbers_match(predicted, gold, tol=0.05):
        return True, "numeric_loose"
    return False, "wrong"


def load_questions(company_filter: str = None, limit: int = 0) -> list:
    """Load JSONL files from eval/datasets/."""
    questions = []
    pattern = f"{company_filter.lower()}*.jsonl" if company_filter else "*.jsonl"
    files = sorted(DATASETS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No JSONL files in {DATASETS_DIR}. "
            f"Pattern: {pattern}"
        )
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                q = json.loads(line)
                q["_file"] = f.name
                questions.append(q)
            except json.JSONDecodeError as e:
                logger.warning("Bad JSON in %s: %s", f.name, e)
    if limit > 0:
        questions = questions[:limit]
    return questions


def run_eval(args) -> list:
    questions = load_questions(args.company, args.limit)
    logger.info("Loaded %d questions from %d file(s)", len(questions),
                len(set(q["_file"] for q in questions)))

    # Group by company
    by_company = defaultdict(list)
    for q in questions:
        by_company[q["company"]].append(q)

    results = []
    t_start = time.time()
    output_path = RESULTS_DIR / f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    for c_idx, (company, q_list) in enumerate(by_company.items(), 1):
        # Find the document
        ticker_map = {
            "Apple Inc.": "apple",
            "Microsoft": "microsoft",
            "NVIDIA": "nvidia",
            "Amazon": "amazon",
            "Alphabet": "alphabet",
            "Meta Platforms": "meta",
            "Tesla": "tesla",
        }
        ticker = ticker_map.get(company, company.lower())
        doc_file = COMPANY_DOCS.get(ticker)
        if not doc_file:
            logger.warning("No doc mapping for %s, skipping", company)
            continue
        doc_path = DOCS_DIR / doc_file
        if not doc_path.exists():
            logger.warning("Missing doc: %s", doc_path)
            for q in q_list:
                results.append({
                    "qid": q["qid"], "question": q["question"][:200],
                    "gold": q["answer"], "predicted": "",
                    "error": "doc_missing", "correct": False,
                })
            continue

        print()
        print("=" * 80)
        print(f" [{c_idx}/{len(by_company)}] {company} — {doc_file} ({len(q_list)} Q)")
        print("=" * 80)

        # Ingest once per company
        t0 = time.time()
        try:
            pipeline = FinBenchPipeline()
            state = pipeline.ingest(
                document_path=str(doc_path),
                session_id=f"custom-{ticker}-{c_idx}",
                company_name=company,
                doc_type=q_list[0].get("doc_type", "10-K"),
                fiscal_year=q_list[0].get("fiscal_year", "FY2023"),
            )
            print(f"   Ingest: {time.time()-t0:.1f}s | "
                  f"chunks={state.chunk_count} | cells={len(state.table_cells)}")
        except Exception as exc:
            print(f"   Ingest FAILED: {exc}")
            for q in q_list:
                results.append({
                    "qid": q["qid"], "question": q["question"][:200],
                    "gold": q["answer"], "predicted": "",
                    "error": f"ingest: {exc}", "correct": False,
                })
            continue

        # Run questions
        for q in q_list:
            t0 = time.time()
            try:
                state = pipeline.query(state, q["question"])
                elapsed = time.time() - t0
                pred = str(getattr(state, "final_answer", "") or "")
                correct, match_type = is_correct(pred, q["answer"])
                pod = getattr(state, "winning_pod", "?") or "?"
                results.append({
                    "qid": q["qid"], "company": company,
                    "question": q["question"][:200], "gold": q["answer"],
                    "unit": q.get("unit", ""), "category": q.get("category", ""),
                    "difficulty": q.get("difficulty", ""),
                    "predicted": pred[:400], "winning_pod": pod,
                    "correct": correct, "match_type": match_type,
                    "elapsed_sec": round(elapsed, 1),
                })
                mark = "✓" if correct else "✗"
                print(f"   {mark} {q['qid']:<10} ({match_type:>14}) "
                      f"{elapsed:>5.1f}s  {pod[:16]:<16}  pred={pred[:50]}")
            except Exception as exc:
                print(f"   ✗ {q.get('qid', '?'):<10} ERROR: {exc}")
                results.append({
                    "qid": q.get("qid", "?"), "company": company,
                    "question": q.get("question", "")[:200],
                    "gold": q.get("answer", ""),
                    "error": str(exc)[:300], "correct": False,
                    "match_type": "exception",
                })

        # Save partial after each company
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - t_start
    print()
    print("=" * 80)
    print(f"  EVAL COMPLETE: {len(results)} questions in {total_time/60:.1f} min")
    print(f"  Saved: {output_path}")
    print("=" * 80)
    return results


def write_summary(results: list) -> Path:
    n = len(results)
    if n == 0: return None
    correct = sum(1 for r in results if r.get("correct"))
    refused = sum(1 for r in results if r.get("match_type") == "refused")
    errors  = sum(1 for r in results if "error" in r and not r.get("correct"))

    summary_path = RESULTS_DIR / f"custom_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    lines = []
    lines.append("# FinBench Custom Eval Results")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Questions:** {n}")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- ✅ Correct: {correct}/{n}  ({100*correct/n:.1f}%)")
    lines.append(f"- ⚠ Refused: {refused}")
    lines.append(f"- ✗ Errors:  {errors}")
    lines.append("")

    # Per-company breakdown
    by_company = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        c = r.get("company", "?")
        by_company[c]["total"] += 1
        if r.get("correct"):
            by_company[c]["correct"] += 1

    lines.append("## Per-Company Breakdown")
    lines.append("")
    lines.append("| Company | Correct | Total | Accuracy |")
    lines.append("|---------|---------|-------|----------|")
    for c, stats in sorted(by_company.items()):
        pct = 100 * stats["correct"] / max(stats["total"], 1)
        lines.append(f"| {c} | {stats['correct']} | {stats['total']} | {pct:.1f}% |")
    lines.append("")

    # Match type breakdown
    by_match = defaultdict(int)
    for r in results:
        by_match[r.get("match_type", "?")] += 1
    lines.append("## Match Type Distribution")
    for m, c in sorted(by_match.items(), key=lambda x: -x[1]):
        lines.append(f"- {m}: {c}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Summary: {summary_path}")
    print("\n".join(lines))
    return summary_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--company", type=str, default=None,
                   help="Filter by company prefix (e.g., 'apple')")
    p.add_argument("--limit", type=int, default=0,
                   help="Max questions (0=all)")
    args = p.parse_args()
    print(f"FinBench Custom Eval | company={args.company or 'all'} | limit={args.limit or 'all'}")
    results = run_eval(args)
    write_summary(results)


if __name__ == "__main__":
    main()