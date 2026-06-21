"""
diag_ap.py - Why does Amazon AP stay 7175 not 34616?
Show every structured_tables row matching accounts_payable + how it parses.
Run: python diag_ap.py
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
for m in list(sys.modules):
    if m.startswith("question_lib"):
        del sys.modules[m]
from src.ingestion.pdf_ingestor import run_pdf_ingestor
from src.state.ba_state import BAState
from question_lib.table_normalizer import _norm, _row_metric, _leading_value_run, _detect_year_columns

cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
hit = next(f for f in cands if "AMAZON_2017" in os.path.basename(f).upper())
st = BAState(document_path=hit, company_name="Amazon", fiscal_year="FY2017", doc_type="10-K")
st = run_pdf_ingestor(st)
stabs = getattr(st,"structured_tables",[])

print("All structured rows whose label maps to accounts_payable:\n")
for tbl in stabs:
    rows = list(tbl.get("rows",[]) or [])
    hdr = tbl.get("headers",[]) or []
    allr = ([hdr]+rows) if hdr else rows
    cy = _detect_year_columns(tbl)
    for row in allr:
        if not row: continue
        if _row_metric(str(row[0])) == "accounts_payable":
            run = _leading_value_run(row)
            print(f"  page{tbl.get('page')} cy={cy}")
            print(f"    RAW ROW: {[str(c) for c in row][:10]}")
            print(f"    leading_run: {run}")
            print()
