"""
diag_amazon_dpo.py - The Amazon AP row is [4294,5030,7175] OLDEST-first.
Verify the year mapping + recompute DPO by hand to find the real bug.
Gold DPO = 93.86.
Run: python diag_amazon_dpo.py
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
for m in list(sys.modules):
    if m.startswith("question_lib"):
        del sys.modules[m]
from src.ingestion.pdf_ingestor import run_pdf_ingestor
from src.state.ba_state import BAState
from question_lib import table_normalizer as t

cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
hit = next(f for f in cands if "AMAZON_2017" in os.path.basename(f).upper())
st = BAState(document_path=hit, company_name="Amazon", fiscal_year="FY2017", doc_type="10-K")
st = run_pdf_ingestor(st)
norm = t.build_normalized(getattr(st,"structured_tables",[]), doc_fiscal_year="FY2017")

print("normalized operands:")
for k in ["accounts_payable","cogs","inventory"]:
    print(f"  {k}: ", {kk[1]:vv for kk,vv in norm.items() if kk[0]==k})

ap17=norm.get(("accounts_payable","2017")); ap16=norm.get(("accounts_payable","2016"))
cogs17=norm.get(("cogs","2017"))
inv17=norm.get(("inventory","2017")); inv16=norm.get(("inventory","2016"))
print(f"\nAP2017={ap17} AP2016={ap16} COGS2017={cogs17} INV2017={inv17} INV2016={inv16}")
if all(x is not None for x in [ap17,ap16,cogs17,inv17,inv16]):
    avg_ap=(ap17+ap16)/2
    denom=cogs17+(inv17-inv16)
    dpo=365*avg_ap/denom
    print(f"DPO = 365*{avg_ap}/{denom} = {dpo:.2f}   (gold 93.86)")
print("\nReal Amazon FY2017 (10-K): AP=34616, COGS=111934, INV2017=16047, INV2016=11461")
print("Real DPO = 365*avg(34616,?)/(111934+(16047-11461))")
