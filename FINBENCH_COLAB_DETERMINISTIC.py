# FinBench — Colab Deterministic Eval (LLM-SKIPPED, full 150 questions)
# Run each cell in order. Sniper-only mode = NO Ollama needed, pure CPU, fast.
# This is the HONEST deterministic-engine score with the whole pipeline wired.

# ============================================================
# CELL 1 — clone your repo + the FinanceBench dataset
# ============================================================
# Replace YOUR_GITHUB_URL with your repo (or upload the folder to Drive).
!git clone https://github.com/YOUR_USERNAME/finbench_agent.git /content/finbench_agent || echo "repo exists / using uploaded copy"
%cd /content/finbench_agent
# FinanceBench questions + PDFs (the eval reads from financebench_dataset/financebench/)
!git clone https://github.com/patronus-ai/financebench.git financebench_dataset/financebench || echo "dataset exists"
!ls financebench_dataset/financebench/pdfs | head -5


# ============================================================
# CELL 2 — install dependencies (NO Ollama, NO FlagEmbedding needed in sniper-only)
# ============================================================
# pdfplumber + pymupdf do the extraction; the 8 fina_* libs do the math.
!pip install -q pdfplumber pymupdf rank-bm25 numpy pandas
# install the 8 sibling libs (they live INSIDE the repo for Colab)
import os, subprocess, glob
for lib in ["Fina_Maths_lib","fina_extractor_lib","fina_pattern_lib","fina_format_lib",
            "Fina_Logic_lib","fina_algo_lib","verify_lib_","fina_question_lib"]:
    p = f"/content/finbench_agent/{lib}"
    if os.path.isdir(p):
        subprocess.run(["pip","install","-q","-e",p], check=False)
print("libs installed")


# ============================================================
# CELL 3 — verify the engine imports + the formula self-test hits gold
# ============================================================
%cd /content/finbench_agent/fina_question_lib/src
!python -m question_lib.advanced_formulas
%cd /content/finbench_agent
# Expect: ROA -0.0153, fixed_asset_turnover 24.26, DPO 93.86, revenue 30.8%, op-income 65.4%


# ============================================================
# CELL 4 — RUN THE FULL 150-Q DETERMINISTIC EVAL (LLM SKIPPED)
# ============================================================
# --sniper-only sets SNIPER_ONLY=1 + DISABLE_BGE=1 -> NO LLM, NO Ollama, NO GPU.
# The preflight is auto-skipped under sniper-only, so it can't abort.
# seed=42 for reproducibility (C5). Runs the WHOLE pipeline per question.
!python eval/run_financebench.py --seed 42 --sniper-only


# ============================================================
# CELL 5 — read the summary (accuracy + GOLD/SILVER/DIAMOND triage)
# ============================================================
import glob, json
summ = sorted(glob.glob("eval/results/financebench_summary_*.md"))
if summ:
    print(open(summ[-1]).read())
res = sorted(glob.glob("eval/results/financebench_*.json"))
res = [r for r in res if "summary" not in r and "partial" not in r]
if res:
    data = json.load(open(res[-1]))
    n = len(data); ok = sum(1 for r in data if r.get("correct"))
    print(f"\nFINAL: {ok}/{n} = {100*ok/n:.2f}%  (deterministic, LLM skipped)")
