"""
diag_operands.py - operands for the numeric SILVER after fixes.
Run: python diag_operands.py
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
for m in list(sys.modules):
    if m.startswith("question_lib"):
        del sys.modules[m]
from src.ingestion.pdf_ingestor import run_pdf_ingestor
from src.state.ba_state import BAState
from question_lib import table_normalizer as t

CASES = [
    ("AMAZON_2017","Amazon","FY2017",[("accounts_payable","2017"),("accounts_payable","2016"),("cogs","2017"),("inventory","2017"),("inventory","2016")]),
    ("ACTIVISIONBLIZZARD_2019","Activision","FY2019",[("revenue","2019"),("ppe","2019"),("ppe","2018"),("capex","2019"),("capex","2018"),("capex","2017")]),
    ("AES_2022","AES","FY2022",[("cogs","2022"),("inventory","2022"),("net_income","2022"),("total_assets","2022"),("total_assets","2021")]),
    ("AMCOR_2023","Amcor","FY2023",[("current_assets","2023"),("current_assets","2022"),("inventory","2023"),("inventory","2022"),("current_liabilities","2023"),("current_liabilities","2022")]),
]
for token,company,fy,operands in CASES:
    cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
    hit = next((f for f in cands if token in os.path.basename(f).upper()), None)
    if not hit: continue
    st = BAState(document_path=hit, company_name=company, fiscal_year=fy, doc_type="10-K")
    st = run_pdf_ingestor(st)
    norm = t.build_normalized(getattr(st,"structured_tables",[]), doc_fiscal_year=fy)
    print(f"\n=== {token} ===  doc_years={t._document_year_order(getattr(st,'structured_tables',[]), anchor_year=int(fy[2:]))}")
    for mid,yr in operands:
        print(f"   {mid:20}@{yr}: {norm.get((mid,yr))}")
