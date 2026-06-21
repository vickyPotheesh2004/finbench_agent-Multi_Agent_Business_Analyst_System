"""
diag_pdfplumber.py - WHY does pdfplumber drop AES cost-of-sales and Amcor
inventory rows? Show what BOTH strategies extract on the income-statement /
balance-sheet pages, and whether the data is in raw_text but not in tables.

Run: python diag_pdfplumber.py
"""
import sys, os, glob, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pdfplumber

def find_pdf(token):
    cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
    return next((f for f in cands if token in os.path.basename(f).upper()), None)

def analyze(token, needles):
    path = find_pdf(token)
    if not path:
        print(f"[{token}] not found"); return
    print("\n" + "="*74)
    print(f" {token}  searching for rows containing: {needles}")
    print("="*74)
    with pdfplumber.open(path) as pdf:
        for pnum, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            low = text.lower()
            if not any(n in low for n in needles):
                continue
            # this page mentions our target line
            for n in needles:
                for line in text.split("\n"):
                    if n in line.lower() and re.search(r"\d", line):
                        print(f"\n  [page {pnum}] RAW TEXT LINE: {line.strip()[:90]!r}")
            # what do the table strategies see on THIS page?
            t_lines = page.extract_tables() or []
            t_text = []
            try:
                t_text = page.extract_tables(table_settings={
                    "vertical_strategy":"text","horizontal_strategy":"text",
                    "snap_tolerance":4,"join_tolerance":4,"edge_min_length":3,
                    "min_words_vertical":2,"min_words_horizontal":1}) or []
            except Exception as e:
                print(f"    text-strategy error: {e}")
            print(f"    -> lines-strategy tables: {len(t_lines)} | text-strategy tables: {len(t_text)}")
            # show any table row that contains our needle
            for strat_name, tbls in [("lines",t_lines),("text",t_text)]:
                for tbl in tbls:
                    for row in tbl:
                        rowstr = " ".join(str(c) for c in row if c).lower()
                        if any(nd in rowstr for nd in needles) and re.search(r"\d", rowstr):
                            cleaned = [str(c).strip() for c in row if c and str(c).strip()]
                            print(f"    [{strat_name}] TABLE ROW: {cleaned[:8]}")
            # only need the first couple of matching pages
            break_after = getattr(analyze, "_count", 0)

analyze("AES_2022", ["cost of sales", "cost of goods"])
analyze("AMCOR_2023", ["inventories", "inventory"])
analyze("AMAZON_2017", ["accounts payable"])
print("\n" + "="*74)
print(" If RAW TEXT has the line but TABLE ROW is missing/garbled -> pdfplumber")
print(" table strategy failed but the data IS in text -> we can parse text rows.")
print("="*74)
