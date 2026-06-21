"""
diag_numeric4.py - Trace the exact wrong operands for the numeric SILVER:
  AES cogs (need ~10000, got 504), Amazon AP (need ~34616, got 7175),
  Amcor inventory/CA/CL, Activision ppe + capex per year.
Shows EVERY matching row so we can pick the right one.

Run: python diag_numeric4.py
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
for m in list(sys.modules):
    if m.startswith("question_lib"):
        del sys.modules[m]

from src.ingestion.pdf_ingestor import run_pdf_ingestor
from src.state.ba_state import BAState
from question_lib.table_normalizer import _norm, _row_metric, _leading_value_run

def scan(token, company, fy, label_keywords):
    cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
    hit = next((f for f in cands if token in os.path.basename(f).upper()), None)
    if not hit:
        print(f"[{token}] not found"); return
    st = BAState(document_path=hit, company_name=company, fiscal_year=fy, doc_type="10-K")
    st = run_pdf_ingestor(st)
    stabs = getattr(st,"structured_tables",[])
    print(f"\n{'='*70}\n {token}\n{'='*70}")
    shown = 0
    for tbl in stabs:
        rows = list(tbl.get("rows",[]) or [])
        hdr = tbl.get("headers",[]) or []
        allr = ([hdr]+rows) if hdr else rows
        for row in allr:
            if not row: continue
            label = str(row[0]); nlabel = _norm(label)
            if any(k in nlabel for k in label_keywords):
                mid = _row_metric(label)
                nums = _leading_value_run(row)
                if nums:  # only rows with numbers
                    print(f"  metric={str(mid):20} | {label[:42]!r:44} nums={nums[:4]}")
                    shown += 1
                    if shown >= 15: break
        if shown >= 15: break

# AES: real cost of sales is ~10,000+ (utility). What rows match 'cost'?
scan("AES_2022","AES","FY2022", ["cost of sales","cost of goods","cost","operating cost"])
# Amazon: real AP ~34,616. What rows match 'payable'?
scan("AMAZON_2017","Amazon","FY2017", ["accounts payable","payable","trade payable"])
# Amcor: inventory + current assets/liabilities
scan("AMCOR_2023","Amcor","FY2023", ["inventor","total current"])
