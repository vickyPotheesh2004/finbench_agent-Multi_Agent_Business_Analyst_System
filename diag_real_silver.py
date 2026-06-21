"""
diag_real_silver.py - Load the REAL FinanceBench questions + gold answers,
run each through the deterministic question_lib path, and classify:
  PASS    - our answer matches gold
  SILVER  - we produced a WRONG value (the bug class we're fixing)
  ABSTAIN - we honestly missed (no confident lie)
Shows the REAL question text + REAL gold so we fix actual targets.

Run:  python diag_real_silver.py
"""
import sys, os, glob, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
for m in list(sys.modules):
    if m.startswith("question_lib"):
        del sys.modules[m]

from src.ingestion.pdf_ingestor import run_pdf_ingestor
from src.state.ba_state import BAState
from question_lib import answer_question

DATA = "financebench_dataset/financebench/data/financebench_open_source.jsonl"
PDFDIR = "financebench_dataset/financebench/pdfs"

# Load all questions
rows = []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"Loaded {len(rows)} FinanceBench questions")

# Match the eval's first 16 docs / 30 questions: the eval iterates docs
# alphabetically and takes the questions per doc. We'll just take the first
# 30 questions grouped by doc_name to mirror the run.
def find_pdf(doc_name):
    for f in glob.glob(os.path.join(PDFDIR, "*.pdf")):
        if doc_name.upper() in os.path.basename(f).upper():
            return f
    # try loose match
    base = doc_name.upper().replace(" ", "")
    for f in glob.glob(os.path.join(PDFDIR, "*.pdf")):
        if base in os.path.basename(f).upper().replace(" ", ""):
            return f
    return None

def norm_num(s):
    """Extract a comparable number from a string, or None."""
    if s is None: return None
    m = re.search(r"-?\$?\s*([\d,]+(?:\.\d+)?)\s*%?", str(s))
    if not m: return None
    try: return float(m.group(1).replace(",", ""))
    except: return None

# Take first 30 questions, group ingestion by doc
rows = rows[:30]
ingest_cache = {}
results = []

for i, r in enumerate(rows, 1):
    q = r.get("question", "")
    gold = r.get("answer", "")
    doc_name = r.get("doc_name", "")
    company = r.get("company", "")
    pdf = find_pdf(doc_name)
    if not pdf:
        results.append(("NOPDF", q[:50], gold[:30], "-"))
        continue
    if pdf not in ingest_cache:
        st = BAState(document_path=pdf, company_name=company)
        st = run_pdf_ingestor(st)
        ingest_cache[pdf] = st
    st = ingest_cache[pdf]

    try:
        res = answer_question(
            q,
            cells=getattr(st,"table_cells",[]),
            raw_text=getattr(st,"raw_text",""),
            company=company,
            fiscal_year=getattr(st,"fiscal_year",""),
            doc_type=getattr(st,"doc_type",""),
            structured_tables=getattr(st,"structured_tables",[]),
        )
        ans = res.final_answer if res.answered else None
    except Exception as e:
        ans = f"ERROR:{e}"

    # classify
    if ans is None:
        cls = "ABSTAIN"
    else:
        gv, av = norm_num(gold), norm_num(ans)
        if gv is not None and av is not None and abs(gv-av) <= max(0.05, abs(gv)*0.02):
            cls = "PASS"
        else:
            cls = "SILVER"   # produced a value that doesn't match gold
    results.append((cls, q[:55], str(gold)[:35], str(ans)[:45] if ans else "ABSTAIN"))

print("\n" + "="*100)
print(f"{'CLASS':8} {'QUESTION':56} {'GOLD':36} OURS")
print("="*100)
counts = {}
for cls, q, gold, ans in results:
    counts[cls] = counts.get(cls,0)+1
    print(f"{cls:8} {q:56} {gold:36} {ans}")

print("\n" + "="*100)
print("COUNTS:", counts)
print("SILVER = wrong value (fix target).  ABSTAIN = honest miss.  PASS = correct.")
print("="*100)
