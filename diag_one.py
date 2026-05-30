"""
diag_one.py — trace ONE FinanceBench question through the pipeline
to find exactly where it breaks.
"""
import os, glob, json
os.environ["SNIPER_ONLY"] = "1"

from src.pipeline.pipeline import FinBenchPipeline

# --- find the 3M 2022 PDF (or any PDF you have) ---
candidates = glob.glob("financebench_dataset/financebench/pdfs/*3M*2022*.pdf")
if not candidates:
    candidates = glob.glob("financebench_dataset/financebench/pdfs/*.pdf")
pdf = candidates[0]
print("PDF:", pdf)

pipe = FinBenchPipeline()

# 1) INGEST — check extraction
state = pipe.ingest(document_path=pdf, company_name="3M", fiscal_year="FY2022")
print("\n=== INGEST RESULTS ===")
print("chunk_count :", getattr(state, "chunk_count", "?"))
print("raw_text len:", len(getattr(state, "raw_text", "") or ""))
print("table_cells :", len(getattr(state, "table_cells", []) or []))

cells = getattr(state, "table_cells", []) or []
print("\n=== FIRST 8 TABLE CELLS (what Sniper/router see) ===")
for c in cells[:8]:
    print(f"  row='{c.get('row_header','')}' | col='{c.get('col_header','')}' | val='{c.get('value','')}'")

# 2) ask a clean formula question
q = "What was 3M's gross margin in FY2022?"
state.query = q
print("\n=== QUESTION ===")
print(q)

# 3) run formula router directly
from src.routing.formula_router import run_formula_router, _detect_formula, _find_input
print("\n=== FORMULA ROUTER TRACE ===")
fid = _detect_formula(q)
print("detected formula:", fid)
if fid:
    import maths_lib as ml
    needed = ml.FORMULA_METADATA_REGISTRY[fid]["inputs"]
    print("formula needs inputs:", needed)
    for inp in needed:
        v = _find_input(cells, inp)
        print(f"  extract '{inp}':", v)

state = run_formula_router(state)
print("\nformula_hit   :", getattr(state, "formula_hit", "?"))
print("formula_answer:", getattr(state, "formula_answer", "?"))

# 4) check sniper too
print("\n=== SNIPER STATE ===")
print("sniper_hit       :", getattr(state, "sniper_hit", "?"))
print("sniper_confidence:", getattr(state, "sniper_confidence", "?"))