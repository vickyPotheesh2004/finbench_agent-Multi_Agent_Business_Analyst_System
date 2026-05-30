"""
fix_shap_bounds.py — robust SHAP index-out-of-bounds fixer
Finds whatever buggy pattern exists and adds the chunk_shap bound.
Run from repo root:  python fix_shap_bounds.py
"""
from pathlib import Path
import re, sys

p = Path("src/analysis/shap_dag.py")
if not p.exists():
    print("ERROR: src/analysis/shap_dag.py not found. Run from repo root.")
    sys.exit(1)

txt = p.read_text(encoding="utf-8")
orig = txt
changed = []

# Pattern 1: the chunk loop guarded only by len(capped)
if "if i < len(chunk_shap) and i < len(capped)" not in txt:
    # match  "for i in top_chunk_idx:\n  <indent> if i < len(capped):"
    m = re.search(r"(for i in top_chunk_idx:\s*\n(\s+))if i < len\(capped\):", txt)
    if m:
        indent = m.group(2)
        txt = txt.replace(
            m.group(0),
            f"{m.group(1)}# FIX(P0): bound by both arrays\n{indent}if i < len(chunk_shap) and i < len(capped):",
            1,
        )
        changed.append("chunk loop bound")

# Pattern 2: feature_names indexing without bound
m2 = re.search(r'"feature":\s*str\(feature_names\[i\]\)(?!\s*if)', txt)
if m2:
    txt = txt.replace(
        m2.group(0),
        '"feature":    str(feature_names[i]) if i < len(feature_names) else str(i)',
        1,
    )
    changed.append("feature_names bound")

# Pattern 3: generic — wrap any abs_shap argsort top_idxs loop
# (only if a raw feature_names[i] still slips through)

if txt != orig:
    p.write_text(txt, encoding="utf-8")
    print("PATCHED shap_dag.py:", ", ".join(changed))
else:
    if "if i < len(chunk_shap) and i < len(capped)" in txt:
        print("Already patched — chunk_shap bound present.")
    else:
        print("No known buggy pattern found. Paste lines 95-125 of")
        print("src/analysis/shap_dag.py so I can patch the exact code.")

# Verify it parses
import ast
try:
    ast.parse(p.read_text(encoding="utf-8-sig"))
    print("Syntax OK")
except SyntaxError as e:
    print("SYNTAX ERROR after patch:", e)
