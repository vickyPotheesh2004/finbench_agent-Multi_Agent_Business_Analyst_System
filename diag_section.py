"""
diag_section.py - Can we detect WHICH financial statement a table belongs to?
Show the section heading nearest each target row, so we can prefer
balance-sheet / income-statement rows over MD&A distractors.

Run: python diag_section.py
"""
import sys, os, glob, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pdfplumber

STATEMENT_MARKERS = [
    "consolidated balance sheet", "consolidated statements of operations",
    "consolidated statements of income", "consolidated statement of financial position",
    "consolidated statements of cash flow", "balance sheet", "statements of operations",
    "income statement", "statement of financial position", "cash flows",
]
DISTRACTOR_MARKERS = [
    "management's discussion", "md&a", "notes to", "risk factors",
    "quantitative and qualitative",
]

def find_pdf(token):
    cands = glob.glob(os.path.join("financebench_dataset","**","*.pdf"), recursive=True)
    return next((f for f in cands if token in os.path.basename(f).upper()), None)

def section_of(text_before):
    """Most recent statement/distractor marker in the text before a row."""
    low = text_before.lower()
    best_pos, best_label = -1, "?"
    for mk in STATEMENT_MARKERS:
        p = low.rfind(mk)
        if p > best_pos:
            best_pos, best_label = p, f"STATEMENT:{mk}"
    for mk in DISTRACTOR_MARKERS:
        p = low.rfind(mk)
        if p > best_pos:
            best_pos, best_label = p, f"DISTRACTOR:{mk}"
    return best_label

def analyze(token, needle):
    path = find_pdf(token)
    print("\n" + "="*70 + f"\n {token}: '{needle}'\n" + "="*70)
    with pdfplumber.open(path) as pdf:
        for pnum, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            low = text.lower()
            if needle not in low:
                continue
            # for each occurrence, what section does the page belong to?
            page_section = section_of(text)
            for line in text.split("\n"):
                if needle in line.lower() and re.search(r"\d", line):
                    print(f"  [pg {pnum}] section={page_section}")
                    print(f"           ROW: {line.strip()[:75]!r}")

analyze("AES_2022", "cost of sales")
analyze("AMAZON_2017", "accounts payable")
analyze("AMCOR_2023", "inventories")
print("\n" + "="*70)
print(" If real rows are on STATEMENT: pages and distractors on DISTRACTOR:")
print(" pages -> we can score by section to always pick the right row.")
print("="*70)
