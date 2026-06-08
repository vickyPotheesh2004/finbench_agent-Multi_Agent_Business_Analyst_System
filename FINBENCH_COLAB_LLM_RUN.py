# ════════════════════════════════════════════════════════════════════════════
#  FINBENCH — COLAB T4 LLM EVAL (the REAL run — no SKIP_LLM)
#  Runs Llama 3.1 8B on a free T4 GPU at ~15 sec/call instead of 3 min/call.
#  All 8 libs + idea-1 (briefing/tools) + idea-2 (intent classifier) ACTIVE.
# ════════════════════════════════════════════════════════════════════════════
#
#  HOW TO USE:
#  1. Go to https://colab.research.google.com  → New notebook
#  2. Runtime → Change runtime type → T4 GPU → Save
#  3. Copy each CELL below into a separate Colab cell, run top to bottom
#  4. Push your repo + 8 libs to GitHub first (see CELL 2 — replace the URLs)
#
#  Each numbered block = one Colab cell.
# ════════════════════════════════════════════════════════════════════════════


# ─── CELL 1 — confirm GPU ────────────────────────────────────────────────────
!nvidia-smi
# You MUST see "Tesla T4" here. If "command not found" → Runtime → Change
# runtime type → T4 GPU. Without GPU this whole thing is pointless.


# ─── CELL 2 — clone your repo + all 8 libs from GitHub ───────────────────────
# FIRST push everything to GitHub from your PC:
#   cd D:\projects\finbench_agent ; git add . ; git commit -m "colab" ; git push
#   (and push each of the 8 lib repos too, OR vendor them — see note below)
#
# Replace <YOUR_GITHUB> with your username/repo.
%cd /content
!git clone https://github.com/<YOUR_GITHUB>/finbench_agent.git
# If your 8 libs are separate repos, clone each:
# !git clone https://github.com/<YOUR_GITHUB>/maths_lib.git
# !git clone https://github.com/<YOUR_GITHUB>/fina_extractor_lib.git
# ... etc for all 8
#
# EASIER OPTION: zip the libs on your PC and upload via Colab's file panel,
# then unzip here:
#   from google.colab import files; files.upload()   # upload libs.zip
#   !unzip -q libs.zip -d /content/libs


# ─── CELL 3 — mount Drive (so results survive disconnect) ────────────────────
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/finbench_results


# ─── CELL 4 — install Ollama ─────────────────────────────────────────────────
!curl -fsSL https://ollama.com/install.sh | sh


# ─── CELL 5 — start Ollama server in the background ──────────────────────────
import subprocess, time, os
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
subprocess.Popen(["ollama", "serve"])
time.sleep(8)
print("Ollama server started")


# ─── CELL 6 — pull Llama 3.1 8B (the C3 model) ───────────────────────────────
!ollama pull llama3.1:8b
# ~4.7 GB download. On T4 this runs at ~15 sec per generation (vs 3 min on CPU).


# ─── CELL 7 — install Python deps + your 8 libs (editable) ───────────────────
%cd /content/finbench_agent
!pip install -q -r requirements.txt 2>/dev/null || echo "no requirements.txt — installing core deps"
!pip install -q bm25s pdfplumber sentence-transformers chromadb rapidfuzz \
    "numpy>=1.26,<2.3" "pandas>=2.2,<2.3" "scipy>=1.13,<1.14" \
    rank-bm25 jinja2 pydantic scikit-learn xgboost ollama jinja2 fitz pymupdf
# Install your 8 libs editable (adjust paths to where you cloned/unzipped them):
for lib in ["maths_lib","fina_extractor_lib","fina_pattern_lib","fina_format_lib",
            "Fina_Logic_lib","fina_algo_lib","verify_lib_","fina_question_lib"]:
    !pip install -q -e /content/libs/{lib} 2>/dev/null || echo "skip {lib}"


# ─── CELL 8 — point pipeline at Ollama + verify libs load ────────────────────
import os
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
os.environ.pop("SKIP_LLM", None)        # ← CRITICAL: LLM is ON
os.environ.pop("SNIPER_ONLY", None)
%cd /content/finbench_agent
!python -m src.utils.lib_bridge        # should show 7/7 installed
!python -c "from src.analysis.llm_briefing import build_briefing; print('briefing OK')"
!python -c "from src.utils.llm_client import OllamaClient; c=OllamaClient(); print('LLM available:', c.is_available())"
# LLM available: True  ← must say True before proceeding


# ─── CELL 9 — SMOKE TEST: 5 questions WITH the LLM ───────────────────────────
# This is the moment of truth — narrative questions get the LLM + briefing.
!python eval/run_financebench.py --seed 42 --limit 5
# Watch: narrative questions (Q4/Q5) should now get REAL answers, not RETRIEVAL_MISS.
# Each question ~30-90 sec (LLM running). 5 questions ≈ 5-8 min.


# ─── CELL 10 — FULL 150-QUESTION RUN (the real benchmark number) ─────────────
# Only run this after CELL 9 looks good. Takes ~4-6 hours on T4.
# Results autosave to Drive every question (survives disconnect).
!python eval/run_financebench.py --seed 42
# When done, the summary .md in /content/drive/MyDrive/finbench_results/
# has your REAL confirmed FinanceBench accuracy. THAT is your HuggingFace number.


# ─── CELL 11 — show the final score ──────────────────────────────────────────
import glob, json
results = sorted(glob.glob('/content/drive/MyDrive/finbench_results/financebench_2*.json'))
if results:
    data = json.load(open(results[-1]))
    n = len(data); c = sum(1 for r in data if r.get('correct'))
    print(f"\n{'='*50}\nFINAL: {c}/{n} = {100*c/n:.1f}% on FinanceBench\n{'='*50}")
    # breakdown by pod
    from collections import Counter
    pods = Counter(r.get('winning_pod','?') for r in data if r.get('correct'))
    print("Correct answers by pod:", dict(pods))
