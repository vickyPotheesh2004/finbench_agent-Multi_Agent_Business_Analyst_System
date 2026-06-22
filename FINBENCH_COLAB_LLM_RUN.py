# ============================================================================
# FinBench — Colab FULL eval WITH LLM (Ollama on)  — the run that actually scores
# Each cell below is one Colab cell. Run top to bottom on a T4 GPU runtime.
# This UPDATES your repo (git pull), re-zips/installs the 8 libs, starts
# Ollama + pulls llama3.1:8b, then runs all 150 with the LLM live.
# ============================================================================

# ─────────────────────────────────────────────────────────────────────────
# CELL 1 — clone OR update the repo (git commands up front, like before)
# ─────────────────────────────────────────────────────────────────────────
import os
%cd /content
if os.path.isdir('/content/finbench_agent/.git'):
    %cd /content/finbench_agent
    !git fetch origin && git reset --hard origin/main && git log --oneline -3
else:
    !git clone https://github.com/vickyPotheesh2004/finbench_agent-Multi_Agent_Business_Analyst_System.git finbench_agent
    %cd /content/finbench_agent
    !git log --oneline -3
# dataset (questions + 150 PDFs)
!test -d financebench_dataset/financebench || git clone https://github.com/patronus-ai/financebench.git financebench_dataset/financebench
print("repo + dataset ready")


# ─────────────────────────────────────────────────────────────────────────
# CELL 2 — mount Drive (for autosaved results) + results dir
# ─────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/finbench_results
print("drive mounted")


# ─────────────────────────────────────────────────────────────────────────
# CELL 3 — upload the 8-lib zip (finbench_libs.zip) and unpack
#   (build the zip on your PC:  the 8 fina_* / *_lib folders at repo root)
#   If the libs are ALREADY committed in the repo, SKIP this cell.
# ─────────────────────────────────────────────────────────────────────────
import os
if not os.path.isdir('/content/finbench_agent/fina_question_lib'):
    from google.colab import files
    up = files.upload()                         # pick finbench_libs.zip
    !rm -rf /content/libs && mkdir -p /content/libs
    !unzip -q finbench_libs.zip -d /content/finbench_agent
print("libs present:", [d for d in os.listdir('/content/finbench_agent') if d.endswith('_lib') or d.startswith('fina')])


# ─────────────────────────────────────────────────────────────────────────
# CELL 4 — install python deps + the 8 libs (editable)
# ─────────────────────────────────────────────────────────────────────────
%cd /content/finbench_agent
!pip install -q pdfplumber pymupdf rank-bm25 numpy pandas requests FlagEmbedding 2>/dev/null
import subprocess, os
for lib in ["Fina_Maths_lib","fina_extractor_lib","fina_pattern_lib","fina_format_lib",
            "Fina_Logic_lib","fina_algo_lib","verify_lib_","fina_question_lib"]:
    p = f"/content/finbench_agent/{lib}"
    if os.path.isdir(p): subprocess.run(["pip","install","-q","-e",p], check=False)
print("deps + libs installed")


# ─────────────────────────────────────────────────────────────────────────
# CELL 5 — install + START Ollama, pull llama3.1:8b  (THE missing piece)
# ─────────────────────────────────────────────────────────────────────────
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time, requests
# start the server in the background
subprocess.Popen(["ollama","serve"])
time.sleep(8)
# pull the EXACT model the code expects (DEFAULT_MODEL = llama3.1:8b)
!ollama pull llama3.1:8b
# verify it answers
for _ in range(10):
    try:
        r = requests.post("http://127.0.0.1:11434/api/chat",
                          json={"model":"llama3.1:8b",
                                "messages":[{"role":"user","content":"say OK"}],
                                "stream":False}, timeout=60)
        print("OLLAMA OK:", r.json().get("message",{}).get("content","")[:40]); break
    except Exception as e:
        print("waiting for ollama...", e); time.sleep(6)


# ─────────────────────────────────────────────────────────────────────────
# CELL 6 — sanity: formula self-test still hits gold (deterministic engine)
# ─────────────────────────────────────────────────────────────────────────
%cd /content/finbench_agent/fina_question_lib/src
!python -m question_lib.advanced_formulas
%cd /content/finbench_agent
# expect: ROA -0.0153, fixed_asset_turnover 24.26, DPO 93.86, revenue 30.8%, op-income 65.4%


# ─────────────────────────────────────────────────────────────────────────
# CELL 7 — RUN ALL 150 WITH THE LLM LIVE  (no --sniper-only this time)
#   The preflight will confirm Ollama+model before starting.
#   ~2-3 hrs on T4. Autosaves to Drive after every question.
# ─────────────────────────────────────────────────────────────────────────
%cd /content/finbench_agent
!python eval/run_financebench.py --seed 42


# ─────────────────────────────────────────────────────────────────────────
# CELL 8 — read the real accuracy + GOLD/SILVER/DIAMOND triage
# ─────────────────────────────────────────────────────────────────────────
import glob, json
summ = sorted(glob.glob("/content/drive/MyDrive/finbench_results/financebench_summary_*.md"))
if summ: print(open(summ[-1]).read())
res = [r for r in sorted(glob.glob("/content/drive/MyDrive/finbench_results/financebench_*.json"))
       if "summary" not in r and "partial" not in r]
if res:
    data = json.load(open(res[-1])); n=len(data); ok=sum(1 for r in data if r.get("correct"))
    print(f"\nFINAL (LLM live): {ok}/{n} = {100*ok/n:.1f}%")
