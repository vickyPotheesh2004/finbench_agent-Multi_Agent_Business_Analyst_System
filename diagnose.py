"""
Deep diagnostic — what does Apple's iXBRL data ACTUALLY look like?
Inspect the cells directly so we stop guessing.
"""
from src.ingestion.pdf_ingestor import PDFIngestor
import re

print("Loading Apple 10-K...")
result = PDFIngestor().ingest("documents/sec_filings/AAPL_FY2023_10-K.html")
cells = result["table_cells"]
print(f"Loaded: {len(cells)} cells\n")

# ── 1. Find all revenue cells ──────────────────────────────────────────
print("=" * 80)
print(" ALL CELLS WITH 'revenue' IN ROW HEADER")
print("=" * 80)
revenue_cells = []
for c in cells:
    rh = (c.get("row_header", "") or "").lower()
    if "revenue" in rh and "cost" not in rh:
        revenue_cells.append(c)

# Sort by value descending
def get_num(c):
    v = c.get("value", "0")
    try:
        return float(re.sub(r"[$,()]", "", str(v).replace(",", "")) or "0")
    except:
        return 0.0

revenue_cells.sort(key=get_num, reverse=True)

print(f"Total revenue cells: {len(revenue_cells)}\n")
print(f"{'Value':<15} {'FY':<10} {'Col Header':<35} {'Row Header':<60}")
print("-" * 130)
for c in revenue_cells[:25]:
    val = str(c.get("value", ""))[:14]
    fy  = str(c.get("fiscal_year", ""))[:9]
    ch  = str(c.get("col_header", ""))[:34]
    rh  = str(c.get("row_header", ""))[:59]
    print(f"{val:<15} {fy:<10} {ch:<35} {rh:<60}")


# ── 2. Find net income cells (most diagnostic — small set) ──────────────
print("\n" + "=" * 80)
print(" ALL us-gaap:NetIncomeLoss CELLS")
print("=" * 80)
ni_cells = [c for c in cells if (c.get("row_header") or "").lower() == "us-gaap:netincomeloss"]
print(f"Total: {len(ni_cells)}\n")
print(f"{'Value':<15} {'FY':<12} {'Col Header':<40} {'Section':<25}")
print("-" * 130)
for c in ni_cells:
    val = str(c.get("value", ""))[:14]
    fy  = str(c.get("fiscal_year", ""))[:11]
    ch  = str(c.get("col_header", ""))[:39]
    sec = str(c.get("section", ""))[:24]
    print(f"{val:<15} {fy:<12} {ch:<40} {sec:<25}")


# ── 3. Sample of total_assets cells ─────────────────────────────────────
print("\n" + "=" * 80)
print(" ALL us-gaap:Assets CELLS")
print("=" * 80)
ta_cells = [c for c in cells if (c.get("row_header") or "").lower() == "us-gaap:assets"]
print(f"Total: {len(ta_cells)}\n")
print(f"{'Value':<15} {'FY':<12} {'Col Header':<40} {'Section':<25}")
print("-" * 130)
for c in ta_cells:
    val = str(c.get("value", ""))[:14]
    fy  = str(c.get("fiscal_year", ""))[:11]
    ch  = str(c.get("col_header", ""))[:39]
    sec = str(c.get("section", ""))[:24]
    print(f"{val:<15} {fy:<12} {ch:<40} {sec:<25}")


# ── 4. Check what fiscal_year values exist ─────────────────────────────
print("\n" + "=" * 80)
print(" UNIQUE FISCAL_YEAR VALUES IN DATASET")
print("=" * 80)
fy_set = set()
for c in cells:
    fy_set.add(str(c.get("fiscal_year", "")))
print(f"Unique values: {sorted(fy_set)}")


# ── 5. Check what col_header patterns exist ────────────────────────────
print("\n" + "=" * 80)
print(" SAMPLE col_header VALUES (first 30 unique)")
print("=" * 80)
ch_set = set()
for c in cells:
    ch = str(c.get("col_header", ""))
    if ch:
        ch_set.add(ch)
        if len(ch_set) >= 30:
            break
for ch in sorted(ch_set):
    print(f"  '{ch}'")