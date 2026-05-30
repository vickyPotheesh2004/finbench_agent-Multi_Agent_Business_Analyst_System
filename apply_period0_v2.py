"""
apply_period0_v2.py — FinBench Period 0 follow-up fixes
═══════════════════════════════════════════════════════════════════
Fixes the 3 issues from the smoke test run:

  FIX A  shap_dag.py : SHAP index out-of-bounds (was never in repo)
  FIX B  pipeline.py : SNIPER_ONLY truly skips LLM pods even on miss
  FIX C  pipeline.py : SNIPER_ONLY miss -> use top retrieval chunk
                        instead of calling slow Llama

Run from repo root:
  python apply_period0_v2.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent.resolve()
applied, skipped, failed = [], [], []


def patch(path_rel, old, new, label):
    p = ROOT / path_rel
    if not p.exists():
        failed.append(f"{label}: file missing {path_rel}"); return
    txt = p.read_text(encoding="utf-8")
    if new.strip() and new in txt:
        skipped.append(f"{label}: already applied"); return
    if old not in txt:
        failed.append(f"{label}: anchor not found"); return
    p.write_text(txt.replace(old, new, 1), encoding="utf-8")
    applied.append(label)


# ── FIX A — SHAP index bounds ───────────────────────────────────────
# The buggy version checks only `i < len(capped)`. Add chunk_shap bound.
patch(
    "src/analysis/shap_dag.py",
    "        for i in top_chunk_idx:\n            if i < len(capped):",
    "        for i in top_chunk_idx:\n            # FIX (P0): bound by BOTH chunk_shap and capped lengths\n            if i < len(chunk_shap) and i < len(capped):",
    "FIX A SHAP bounds",
)

# Some versions use a different variable name for the loop; try a 2nd form
patch(
    "src/analysis/shap_dag.py",
    "top_features = [\n            {\n                \"feature\":    str(feature_names[i]),",
    "top_features = [\n            {\n                \"feature\":    str(feature_names[i]) if i < len(feature_names) else str(i),",
    "FIX A2 feature_names bound",
)


# ── FIX B + C — SNIPER_ONLY hard LLM bypass ─────────────────────────
# When SNIPER_ONLY=1, skip N11-N15 (all LLM pods) entirely.
# If sniper missed, fall back to the top retrieval chunk as the answer.
#
# We insert a bypass block right BEFORE the "N11 Analyst" pod call.

POD_OLD = """        # ──────────────────────────────────────────────────────────────────────
        # Pods
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            \"N11 Analyst\","""

POD_NEW = """        # ──────────────────────────────────────────────────────────────────────
        # SNIPER-ONLY LLM BYPASS (P0)
        # ──────────────────────────────────────────────────────────────────────
        if os.environ.get(\"SNIPER_ONLY\") == \"1\":
            # Skip all LLM pods. Use best available retrieval result.
            _best = \"\"
            for _src in (
                getattr(state, \"reranked_chunks\", None),
                getattr(state, \"retrieval_stage_2\", None),
                getattr(state, \"bm25_results\", None),
            ):
                if _src:
                    _c = _src[0]
                    _best = (
                        _c.get(\"text\")
                        or _c.get(\"page_content\")
                        or \"\"
                    ) if isinstance(_c, dict) else str(_c)
                    if _best:
                        break

            if getattr(state, \"sniper_hit\", False) and getattr(state, \"sniper_answer\", \"\"):
                state.final_answer = state.sniper_answer
                state.final_answer_pre_xgb = state.sniper_answer
                state.confidence_score = getattr(state, \"sniper_confidence\", 0.5)
                state.winning_pod = \"SNIPER_ONLY\"
            else:
                state.final_answer = _best[:500] if _best else \"RETRIEVAL_MISS\"
                state.final_answer_pre_xgb = state.final_answer
                state.confidence_score = 0.1
                state.winning_pod = \"SNIPER_ONLY_RETRIEVAL\"

            state = _safe_run(\"N18 RLEF\", run_rlef_engine, state)
            state = _safe_run(\"N19 Output\", run_output_generator, state)
            cleanup_memory()
            return state

        # ──────────────────────────────────────────────────────────────────────
        # Pods
        # ──────────────────────────────────────────────────────────────────────

        state = _safe_run(
            \"N11 Analyst\","""

patch("src/pipeline/pipeline.py", POD_OLD, POD_NEW, "FIX B+C SNIPER_ONLY bypass")


# ── Report ──────────────────────────────────────────────────────────
print("=" * 60)
print("PERIOD 0 v2 PATCH REPORT")
print("=" * 60)
print("\nAPPLIED:")
for a in applied: print(f"  OK {a}")
print("\nSKIPPED:")
for s in skipped: print(f"  -- {s}")
if failed:
    print("\nMANUAL CHECK NEEDED:")
    for f in failed: print(f"  !! {f}")
print()
if "FIX B+C SNIPER_ONLY bypass" in applied or any("SNIPER" in s for s in skipped):
    print("Next: python eval/run_financebench.py --limit 5 --seed 42 --sniper-only")
    print("Expected: ~30s/Q, NO LLM timeout warnings, completes fast.")
