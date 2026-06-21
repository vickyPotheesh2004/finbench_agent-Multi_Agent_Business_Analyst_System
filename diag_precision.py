"""
diag_precision.py - ONE check: exact operands for the 3 precision-miss SILVER.
Shows every number feeding each formula vs what the gold needs.
Run: python diag_precision.py
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
    norm = t.build_normalized(getattr(st,"structured_tables",[]), doc_fiscal_year=fy)
    return norm, getattr(st,"raw_text","")

# ---- AMAZON DPO (95.86 vs 93.86) ----
print("="*70); print(" AMAZON DPO  gold 93.86"); print("="*70)
norm, raw = load("AMAZON_2017","Amazon","FY2017")
ap17 = a._operand(norm, raw, "accounts_payable", 2017, anchor_year=2017)
ap16 = a._operand(norm, raw, "accounts_payable", 2016, anchor_year=2017)
cogs = a._operand(norm, raw, "cogs", 2017)
inv17 = norm.get(("inventory","2017")); inv16 = norm.get(("inventory","2016"))
run = a._raw_lookup_max(raw, "accounts_payable", 2017)
print(f"  AP run from raw_text: {run}")
print(f"  AP2017={ap17}  AP2016={ap16}  (gold needs 34616, 25309)")
print(f"  COGS={cogs} (gold 111934)  INV2017={inv17} (16047)  INV2016={inv16} (11461)")
if all(x is not None for x in [ap17,ap16,cogs,inv17,inv16]):
    avg=(ap17+ap16)/2; den=cogs+(inv17-inv16)
    print(f"  DPO=365*{avg}/{den}={365*avg/den:.2f}")
    print(f"  gold operands: 365*29962.5/116520 = {365*29962.5/116520:.2f}")

# ---- ACTIVISION fixed-asset-turnover (25.65 vs 24.26) ----
print("="*70); print(" ACTIVISION fixed-asset-turnover  gold 24.26"); print("="*70)
norm, raw = load("ACTIVISIONBLIZZARD_2019","Activision","FY2019")
rev = norm.get(("revenue","2019"))
ppe19 = a._operand(norm, raw, "ppe", 2019); ppe18 = a._operand(norm, raw, "ppe", 2018)
print(f"  revenue2019={rev} (gold 6489)")
print(f"  PPE2019={ppe19}  PPE2018={ppe18}")
if rev and ppe19 and ppe18:
    print(f"  turnover=6489/avg({ppe19},{ppe18})={rev/((ppe19+ppe18)/2):.2f}")
print(f"  gold: 6489/avg(253,282)=6489/267.5={6489/267.5:.2f}")

# ---- ACTIVISION 3yr-capex (2.1% vs 1.9%) ----
print("="*70); print(" ACTIVISION 3yr-capex%  gold 1.9%"); print("="*70)
for y in (2019,2018,2017):
    cx=norm.get(("capex",str(y))); rv=norm.get(("revenue",str(y)))
    pct = (abs(cx)/rv*100) if (cx and rv) else None
    print(f"  {y}: capex={cx} revenue={rv} -> {pct}")
print("  gold needs capex/rev for 2017,2018,2019 averaging to 1.9%")
