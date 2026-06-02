"""
diagnose_baseline.py — honest FinanceBench baseline + per-question failure map.
Runs the CURRENT pipeline (maths_lib wired, no new libs) and logs WHY each
question passes or fails, bucketed by failure cause.

Usage:
  python diagnose_baseline.py --limit 20 --seed 42
"""
from __future__ import annotations
import os, json, argparse, glob, time
from pathlib import Path

os.environ.setdefault("SNIPER_ONLY", "1")   # deterministic, no slow LLM
os.environ.setdefault("DISABLE_BGE", "1")

from src.pipeline.pipeline import FinBenchPipeline

# reuse the eval's own grader so numbers match the official run
import importlib.util
spec = importlib.util.spec_from_file_location("rfb", "eval/run_financebench.py")
rfb = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(rfb)
    is_correct = rfb.is_correct
except Exception:
    # fallback grader if import side-effects fail
    def is_correct(pred, gold):
        return (gold.strip().lower() in (pred or "").lower(), "fallback", 0.0)


def load_questions(limit):
    """Load FinanceBench Q/A pairs from the dataset jsonl."""
    qpath = None
    for cand in [
        "financebench_dataset/financebench/data/financebench_open_source.jsonl",
        "financebench_dataset/financebench/financebench_open_source.jsonl",
    ]:
        if os.path.exists(cand):
            qpath = cand
            break
    if not qpath:
        hits = glob.glob("financebench_dataset/**/*.jsonl", recursive=True)
        qpath = hits[0] if hits else None
    if not qpath:
        raise SystemExit("No FinanceBench jsonl found. Check dataset path.")

    rows = []
    with open(qpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def classify_q(q):
    ql = q.lower()
    if any(w in ql for w in ["margin", "ratio", "return on", "turnover", "per share", "growth"]):
        return "ratio"
    if any(w in ql for w in ["why", "explain", "describe", "what drove", "discuss"]):
        return "narrative"
    return "extraction"


def find_pdf(doc_name):
    for ext in ("", ".pdf"):
        p = f"financebench_dataset/financebench/pdfs/{doc_name}{ext}"
        if os.path.exists(p):
            return p
    hits = glob.glob(f"financebench_dataset/financebench/pdfs/*{doc_name}*")
    return hits[0] if hits else None


def failure_bucket(state, correct, qtype, cells):
    if correct:
        return "CORRECT"
    if cells == 0:
        return "FAIL_extraction_empty"          # no cells extracted
    wp = getattr(state, "winning_pod", "")
    if qtype == "narrative":
        return "FAIL_narrative_no_number"
    if getattr(state, "formula_hit", False):
        return "FAIL_formula_wrong_input"        # the 53.8 class
    if wp in ("SNIPER_ONLY_RETRIEVAL", ""):
        return "FAIL_sniper_miss_then_retrieval" # sniper didn't find it
    if getattr(state, "sniper_hit", False):
        return "FAIL_sniper_wrong_value"
    return "FAIL_other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_questions(args.limit)
    pipe = FinBenchPipeline()

    results = []
    buckets = {}
    correct_n = 0
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        q = row.get("question", "")
        gold = str(row.get("answer", ""))
        doc = row.get("doc_name") or row.get("document") or ""
        qtype = classify_q(q)
        pdf = find_pdf(doc)

        print(f"\n[{i}/{len(rows)}] {doc} | {qtype}")
        print(f"  Q: {q[:90]}")

        if not pdf:
            bucket = "FAIL_no_pdf"
            buckets[bucket] = buckets.get(bucket, 0) + 1
            results.append({"q": q, "doc": doc, "bucket": bucket})
            print(f"  -> NO PDF FOUND for {doc}")
            continue

        try:
            state = pipe.ingest(document_path=pdf, company_name=doc.split("_")[0],
                                fiscal_year="")
            state = pipe.query(state, q)
            pred = getattr(state, "final_answer", "") or ""
            cells = len(getattr(state, "table_cells", []) or [])
            ok, mode, score = is_correct(pred, gold)
        except Exception as e:
            pred, cells, ok, mode = f"ERROR: {e}", 0, False, "error"
            state = type("S", (), {})()

        bucket = failure_bucket(state, ok, qtype, cells)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        if ok:
            correct_n += 1

        print(f"  gold: {gold[:60]}")
        print(f"  pred: {pred[:60]}")
        print(f"  cells={cells} pod={getattr(state,'winning_pod','?')} "
              f"formula_hit={getattr(state,'formula_hit','?')} -> {bucket}")

        results.append({
            "q": q, "doc": doc, "qtype": qtype, "gold": gold,
            "pred": pred[:200], "cells": cells,
            "winning_pod": getattr(state, "winning_pod", ""),
            "formula_hit": getattr(state, "formula_hit", False),
            "sniper_hit": getattr(state, "sniper_hit", False),
            "correct": ok, "bucket": bucket,
        })

    # ---- report ----
    n = len(rows)
    acc = correct_n / n * 100 if n else 0
    mins = (time.time() - t0) / 60

    print("\n" + "=" * 60)
    print(f"HONEST BASELINE (maths_lib only)")
    print("=" * 60)
    print(f"Questions: {n}  |  Correct: {correct_n}  |  Accuracy: {acc:.1f}%")
    print(f"Time: {mins:.1f} min")
    print("\nFAILURE BUCKETS (ranked):")
    for b, c in sorted(buckets.items(), key=lambda x: -x[1]):
        print(f"  {c:3d}  {b}")

    Path("eval/results").mkdir(parents=True, exist_ok=True)
    out = f"eval/results/diagnose_baseline_{int(time.time())}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "n": n, "correct": correct_n,
                   "buckets": buckets, "results": results}, f, indent=2)
    print(f"\nSaved: {out}")
    print("\nNEXT: read the top failure bucket — that's where your points are.")


if __name__ == "__main__":
    main()