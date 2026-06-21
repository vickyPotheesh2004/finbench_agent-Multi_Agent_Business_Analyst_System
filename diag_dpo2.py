"""
diag_dpo2.py - Amazon DPO is 95.86 vs gold 93.86. Check the exact operands
the solver uses, esp. AP[2017] vs AP[2016] (should be 34616 vs 25309).
Run: python diag_dpo2.py
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

cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
hit = next(f for f in cands if "AMAZON_2017" in os.path.basename(f).upper())
st = BAState(document_path=hit, company_name="Amazon", fiscal_year="FY2017", doc_type="10-K")
st = run_pdf_ingestor(st)
raw = getattr(st,"raw_text","")
norm = t.build_normalized(getattr(st,"structured_tables",[]), doc_fiscal_year="FY2017")

# what does _raw_lookup_max return as the run?
run = a._raw_lookup_max(raw, "accounts_payable", 2017)
print(f"AP run (newest-first): {run}")
ap17 = a._operand(norm, raw, "accounts_payable", 2017, anchor_year=2017)
ap16 = a._operand(norm, raw, "accounts_payable", 2016, anchor_year=2017)
cogs = a._operand(norm, raw, "cogs", 2017)
inv17 = norm.get(("inventory","2017")); inv16 = norm.get(("inventory","2016"))
print(f"AP2017={ap17}  AP2016={ap16}")
print(f"COGS2017={cogs}  INV2017={inv17} INV2016={inv16}")
if all(x is not None for x in [ap17,ap16,cogs,inv17,inv16]):
    avg_ap=(ap17+ap16)/2; denom=cogs+(inv17-inv16)
    print(f"DPO = 365*{avg_ap}/{denom} = {365*avg_ap/denom:.2f}  (gold 93.86)")
print("\nGold needs: AP2017=34616 AP2016=25309 -> avg=29962.5")
print("If AP2016 also = 34616, that's the 2-pt error.")
