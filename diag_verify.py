"""
diag_verify.py - verify the inventory + ppe + AP raw runs and final operands
after the _PREFER_RAW_MAX changes. Confirm Amazon DPO + Activision turnover.
Run: python diag_verify.py
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
for m in list(sys.modules):
    if m.startswith("question_lib"):
        del sys.modules[m]
from src.ingestion.pdf_ingestor import run_pdf_ingestor
from src.state.ba_state import BAState
from question_lib import table_normalizer as t
from question_lib import advanced_formulas as a

def load(token, company, fy):
    cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
    hit = next(f for f in cands if token in os.path.basename(f).upper())
    st = BAState(document_path=hit, company_name=company, fiscal_year=fy, doc_type="10-K")
    st = run_pdf_ingestor(st)
    return t.build_normalized(getattr(st,"structured_tables",[]), doc_fiscal_year=fy), getattr(st,"raw_text","")

print("AMAZON:")
norm, raw = load("AMAZON_2017","Amazon","FY2017")
print("  AP run:", a._raw_lookup_max(raw,"accounts_payable",2017))
print("  INV run:", a._raw_lookup_max(raw,"inventory",2017))
print("  AP17:", a._operand(norm,raw,"accounts_payable",2017,anchor_year=2017),
      "AP16:", a._operand(norm,raw,"accounts_payable",2016,anchor_year=2017))
print("  INV17:", a._operand(norm,raw,"inventory",2017,anchor_year=2017),
      "INV16:", a._operand(norm,raw,"inventory",2016,anchor_year=2017))
r = a.solve("Amazon FY2017 days payable outstanding DPO", norm, "FY2017", raw_text=raw)
print("  DPO:", r, " (gold 93.86)")

print("ACTIVISION:")
norm, raw = load("ACTIVISIONBLIZZARD_2019","Activision","FY2019")
print("  PPE run:", a._raw_lookup_max(raw,"ppe",2019))
print("  PPE19:", a._operand(norm,raw,"ppe",2019,anchor_year=2019),
      "PPE18:", a._operand(norm,raw,"ppe",2018,anchor_year=2019))
r = a.solve("Activision FY2019 fixed asset turnover ratio", norm, "FY2019", raw_text=raw)
print("  turnover:", r, " (gold 24.26)")

print("AES (must still pass 9.54):")
norm, raw = load("AES_2022","AES","FY2022")
r = a.solve("AES how many times sold inventory turnover FY2022", norm, "FY2022", raw_text=raw)
print("  inv turnover:", r, " (gold 9.5)")
