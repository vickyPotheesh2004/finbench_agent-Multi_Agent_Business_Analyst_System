# FINBENCH_COLAB_EVAL.md
# Run FinBench eval on free Colab T4 GPU.
# Each section = one Colab cell. Run sequentially.

═══════════════════════════════════════════════════════════════════════════
                      OFFICIAL ACCURACY EVAL — COLAB GUIDE
═══════════════════════════════════════════════════════════════════════════

PREREQUISITES:
  - Free Google Colab account
  - Runtime → Change runtime type → GPU (T4)
  - Expected time: 2-4 hours for full FinanceBench (150 Qs)
  - Cost: $0 (free tier)


═════════════════════════════════════════════════════════════════════════
 CELL 1 — Confirm GPU + workspace setup
═════════════════════════════════════════════════════════════════════════
```python
!nvidia-smi
import os, sys
print("Python:", sys.version)
print("GPU:", os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip())
```

═════════════════════════════════════════════════════════════════════════
 CELL 2 — Mount Drive (so eval results persist after Colab times out)
═════════════════════════════════════════════════════════════════════════
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/finbench_eval_results
```

═════════════════════════════════════════════════════════════════════════
 CELL 3 — Clone the FinBench repo + 8 support libs
═════════════════════════════════════════════════════════════════════════
```python
# Replace <yourname> with your actual GitHub handle once published
import subprocess
%cd /content
!git clone https://github.com/<yourname>/finbench_agent.git
%cd finbench_agent

# Clone the 8 support libs
!mkdir -p ../libs
%cd ../libs
!git clone https://github.com/<yourname>/maths_lib_.git
!git clone https://github.com/<yourname>/fina_extractor_lib.git
!git clone https://github.com/<yourname>/fina_pattern_lib.git
!git clone https://github.com/<yourname>/fina_format_lib.git
!git clone https://github.com/<yourname>/Fina_Logic_lib.git
!git clone https://github.com/<yourname>/fina_algo_lib.git
!git clone https://github.com/<yourname>/verify_lib_.git
!git clone https://github.com/<yourname>/fina_question_lib.git
```

═════════════════════════════════════════════════════════════════════════
 CELL 4 — Install Python dependencies
═════════════════════════════════════════════════════════════════════════
```python
%cd /content/finbench_agent
!pip install -q -r requirements.txt
!pip install -q langgraph langchain pydantic chromadb sentence-transformers bm25s pdfplumber rich rank-bm25
```

═════════════════════════════════════════════════════════════════════════
 CELL 5 — Install the 8 support libs (editable)
═════════════════════════════════════════════════════════════════════════
```python
!pip install -e /content/libs/maths_lib_ --quiet
!pip install -e /content/libs/fina_extractor_lib --quiet
!pip install -e /content/libs/fina_pattern_lib/fina_pattern_lib --quiet
!pip install -e /content/libs/fina_format_lib --quiet
!pip install -e /content/libs/Fina_Logic_lib/logic_lib_ --quiet
!pip install -e /content/libs/fina_algo_lib --quiet
!pip install -e /content/libs/verify_lib_ --quiet
!pip install -e /content/libs/fina_question_lib --quiet

# Verify
!python -m src.utils.lib_bridge
```

═════════════════════════════════════════════════════════════════════════
 CELL 6 — Install + start Ollama on Colab GPU
═════════════════════════════════════════════════════════════════════════
```python
# Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh

# Start as background service
import subprocess, time, os, signal
ollama_proc = subprocess.Popen(
    ['ollama', 'serve'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    preexec_fn=os.setsid,
)
time.sleep(5)

# Pull Llama 3.1 8B (one-time, ~4GB download)
!ollama pull llama3.1:8b

# Verify
!ollama list
```

═════════════════════════════════════════════════════════════════════════
 CELL 7 — Set up FinanceBench data
═════════════════════════════════════════════════════════════════════════
```python
# Download FinanceBench from HuggingFace
%cd /content/finbench_agent
!pip install -q datasets

import os
os.makedirs('eval/datasets/financebench', exist_ok=True)

from datasets import load_dataset
ds = load_dataset("PatronusAI/financebench")
ds['train'].to_json('eval/datasets/financebench/financebench_open_source.jsonl')

# Download the actual PDFs from the FinanceBench GitHub repo
!git clone --depth 1 https://github.com/patronus-ai/financebench.git /tmp/fb_pdfs
!cp -r /tmp/fb_pdfs/pdfs/ eval/datasets/financebench/pdfs/
print("PDFs:", len(os.listdir('eval/datasets/financebench/pdfs')))
```

═════════════════════════════════════════════════════════════════════════
 CELL 8 — SMOKE TEST (5 questions, verify pipeline works)
═════════════════════════════════════════════════════════════════════════
```python
%cd /content/finbench_agent
!python eval/run_financebench.py --limit 5 --seed 42 2>&1 | tee /content/drive/MyDrive/finbench_eval_results/smoke_test.log

# If smoke test gets >= 2/5, proceed to full eval
# If smoke test gets 0-1/5, STOP and debug (don't waste 3 hours)
```

═════════════════════════════════════════════════════════════════════════
 CELL 9 — FULL FINANCEBENCH EVAL (the official number, 2-3 hours)
═════════════════════════════════════════════════════════════════════════
```python
# This is THE OFFICIAL RUN that produces the number you put on HuggingFace
%cd /content/finbench_agent
!python eval/run_financebench.py --seed 42 2>&1 | tee /content/drive/MyDrive/finbench_eval_results/financebench_full_$(date +%Y%m%d_%H%M).log

# Save results to Drive (so Colab timeout doesn't lose them)
!cp -r eval/results /content/drive/MyDrive/finbench_eval_results/
!ls -la /content/drive/MyDrive/finbench_eval_results/results/
```

═════════════════════════════════════════════════════════════════════════
 CELL 10 — Generate summary table for HuggingFace model card
═════════════════════════════════════════════════════════════════════════
```python
import json, glob, os
from collections import Counter

# Find the latest result file
result_files = sorted(glob.glob('/content/drive/MyDrive/finbench_eval_results/results/financebench_*.json'))
latest = result_files[-1]
with open(latest) as f:
    data = json.load(f)

# Count outcomes
status_counter = Counter(r.get('status', 'unknown') for r in data['results'])
pod_counter = Counter(r.get('winning_pod', 'none') for r in data['results'])

total = len(data['results'])
correct = sum(1 for r in data['results'] if r.get('status') == 'correct')
accuracy = correct / total * 100 if total else 0

print("=" * 60)
print(f"FinanceBench Official Eval Result (seed=42)")
print("=" * 60)
print(f"Total questions:   {total}")
print(f"Correct:           {correct}")
print(f"Accuracy:          {accuracy:.2f}%")
print()
print(f"Status breakdown:")
for status, count in status_counter.most_common():
    print(f"  {status:<20} {count:>4}  ({count/total*100:.1f}%)")
print()
print(f"Winning pod breakdown:")
for pod, count in pod_counter.most_common():
    print(f"  {pod:<25} {count:>4}  ({count/total*100:.1f}%)")
```

═════════════════════════════════════════════════════════════════════════
 CELL 11 — Run secondary benchmarks (optional, for richer HF card)
═════════════════════════════════════════════════════════════════════════
```python
# FinQA
!python eval/run_finqa.py --limit 100 --seed 42 2>&1 | \
  tee /content/drive/MyDrive/finbench_eval_results/finqa_100.log

# TAT-QA
!python eval/run_tatqa.py --limit 100 --seed 42 2>&1 | \
  tee /content/drive/MyDrive/finbench_eval_results/tatqa_100.log
```

═════════════════════════════════════════════════════════════════════════
 CELL 12 — Push to HuggingFace
═════════════════════════════════════════════════════════════════════════
```python
# Login (will prompt for HF token from huggingface.co/settings/tokens)
!huggingface-cli login

# Create repo + upload
from huggingface_hub import HfApi, create_repo
api = HfApi()

REPO_NAME = "<yourname>/finbench-analyst-v1"
create_repo(REPO_NAME, repo_type="model", exist_ok=True)

# Upload code + model card
api.upload_folder(
    folder_path="/content/finbench_agent",
    repo_id=REPO_NAME,
    repo_type="model",
    ignore_patterns=["__pycache__", "*.pyc", "cache/*", ".git/*"],
)

# Upload eval results so anyone can verify
api.upload_folder(
    folder_path="/content/drive/MyDrive/finbench_eval_results",
    repo_id=REPO_NAME,
    repo_type="model",
    path_in_repo="evaluation_results",
)

print(f"Done. View at: https://huggingface.co/{REPO_NAME}")
```


═══════════════════════════════════════════════════════════════════════════
                  IMPORTANT NOTES ON OFFICIAL EVAL
═══════════════════════════════════════════════════════════════════════════

1. RUN ONCE PER COMMIT
   Every code change requires a full eval re-run if you claim the number.
   Don't keep old numbers from old code — that's misleading.

2. SAVE RAW LOGS
   The /content/drive/MyDrive/finbench_eval_results/ folder should be your
   permanent archive. Companies/reviewers will ask: "show me the raw log".

3. RANDOM-SEED DOCUMENTATION
   Every number on HuggingFace must have seed=42 noted.
   Don't run eval 10 times and report best — report seed=42 only.

4. KNOW THE LIMITS
   FinanceBench has 150 questions. Some are unanswerable from the doc.
   Don't expect 100% — top systems hit ~54%.

5. COMPARABLE OPEN-SOURCE NUMBERS (as of 2026):
   - GPT-4o + RAG (Patronus baseline): 54% on FinanceBench
   - Llama 4 Maverick (no RAG):        39%
   - Open-source RAG (best so far):    ~50-55%
   - YOUR TARGET:                       legitimate 40-55%

If we get 40%+ legitimately, that's competitive open-source.
If we get 50%+, that's noteworthy.
If we get 55%+, that's HF-leaderboard-worthy.
If we get 70%+, I would suspect we trained on test set — DON'T do that.
