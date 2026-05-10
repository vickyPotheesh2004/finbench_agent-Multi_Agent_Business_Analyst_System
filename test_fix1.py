"""Quick test of Fix 1 — Sniper precision on real Apple 10-K."""
from src.ingestion.pdf_ingestor import PDFIngestor
from src.retrieval.sniper_rag import run_sniper

print("Loading Apple 10-K...")
result = PDFIngestor().ingest("documents/sec_filings/AAPL_FY2023_10-K.html")
cells = result["table_cells"]
print(f"Loaded: {len(cells)} cells\n")

QUESTIONS = [
    ("revenue",          "What was Apple total revenue (or net sales) in FY2023?",          "383,285"),
    ("cogs",             "What was Apple cost of revenue (or cost of sales) in FY2023?",    "214,137"),
    ("gross_profit",     "What was Apple gross profit in FY2023?",                          "169,148"),
    ("operating_income", "What was Apple operating income in FY2023?",                      "114,301"),
    ("net_income",       "What was Apple net income in FY2023?",                            "96,995"),
    ("eps_diluted",      "What was Apple diluted earnings per share in FY2023?",            "6.13"),
    ("total_assets",     "What were Apple total assets at the end of FY2023?",              "352,583"),
    ("total_liab",       "What were Apple total liabilities at the end of FY2023?",         "290,437"),
    ("long_term_debt",   "What was Apple long-term debt at the end of FY2023?",             "95,281"),
    ("operating_cf",     "What was Apple net cash provided by operating activities in FY2023?", "110,543"),
    ("free_cash_flow",   "What was Apple free cash flow in FY2023?",                        "99,584"),
    ("deferred_revenue", "What was Apple deferred revenue at the end of FY2023?",           "8,061"),
    ("interest_expense", "What was Apple interest expense in FY2023?",                      "3,933"),
    ("income_tax",       "What was Apple provision for income taxes in FY2023?",            "16,741"),
    ("cash",             "How much cash and cash equivalents did Apple have at the end of FY2023?", "29,965"),
]

print("=" * 80)
print(" FIX 1 VERIFICATION — APPLE FY2023")
print("=" * 80)
print(f"{'#':<3} {'Key':<18} {'Expected':<12} {'Got':<14} {'Match':<6} {'Cell row':<50}")
print("-" * 110)

correct = 0
wrong = 0
miss = 0

for i, (key, q, expected) in enumerate(QUESTIONS, 1):
    r = run_sniper(q, cells)
    
    # Extract numeric from answer
    import re
    nums = re.findall(r'\d{1,3}(?:,\d{3})+|\d+\.\d+', r.answer)
    got_main = nums[0] if nums else "(none)"
    
    if not r.sniper_hit:
        match_str = "✗ MISS"
        miss += 1
    elif got_main.replace(",", "") == expected.replace(",", ""):
        match_str = "✓ HIT"
        correct += 1
    else:
        match_str = "✗ WRONG"
        wrong += 1
    
    cell_row = (r.cell.row_header[:48] if r.cell else "(none)")
    print(f"{i:<3} {key:<18} {expected:<12} {got_main:<14} {match_str:<6} {cell_row:<50}")

print("-" * 110)
print(f"Correct: {correct}/{len(QUESTIONS)} ({100*correct/len(QUESTIONS):.1f}%)")
print(f"Wrong:   {wrong}/{len(QUESTIONS)}")
print(f"Miss:    {miss}/{len(QUESTIONS)}")