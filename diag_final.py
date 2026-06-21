"""
diag_final.py - verify AES COGS, Amazon AP, Amcor after expense+magnitude fixes
Run: python diag_final.py
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
for m in list(sys.modules):
    if m.startswith("question_lib"):
        del sys.modules[m]
from src.ingestion.pdf_ingestor import run_pdf_ingestor
from src.state.ba_state import BAState
from question_lib import table_normalizer as t

for token,company,fy,checks in [
    ("AES_2022","AES","FY2022",[("cogs","2022","~10069"),("cogs","2021","~8430"),("inventory","2022","?")]),
    ("AMAZON_2017","Amazon","FY2017",[("accounts_payable","2017","~34616"),("accounts_payable","2016","~25309"),("cogs","2017","~111934"),("inventory","2017","~16047"),("inventory","2016","~11461")]),
    ("AMCOR_2023","Amcor","FY2023",[("current_assets","2023","~5863"),("inventory","2023","?"),("current_liabilities","2023","~4393")]),
]:
    cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
    hit = next((f for f in cands if token in os.path.basename(f).upper()), None)
    if not hit: continue
    st = BAState(document_path=hit, company_name=company, fiscal_year=fy, doc_type="10-K")
    st = run_pdf_ingestor(st)
    norm = t.build_normalized(getattr(st,"structured_tables",[]), doc_fiscal_year=fy)
    print(f"\n=== {token} ===")
    for mid,yr,expect in checks:
        print(f"   {mid:18}@{yr}: {norm.get((mid,yr))}   (expect {expect})")
