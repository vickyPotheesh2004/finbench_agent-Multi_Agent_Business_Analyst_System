"""
apply_period0.py — FinBench Period 0 UNBLOCK
Applies all runtime fixes. Idempotent (safe to run twice).
"""
from pathlib import Path
import sys, re

ROOT = Path(__file__).parent.resolve()
applied, skipped, failed = [], [], []

def patch(path_rel, old, new, label):
    p = ROOT / path_rel
    if not p.exists():
        failed.append(f"{label}: file missing {path_rel}"); return
    txt = p.read_text(encoding="utf-8")
    if new.strip() and new in txt:
        skipped.append(f"{label}: already applied"); return
    if old not in txt:
        failed.append(f"{label}: anchor not found"); return
    p.write_text(txt.replace(old, new, 1), encoding="utf-8")
    applied.append(label)

# P0.1a — MAX_RETRIES 3 -> 1
patch("src/utils/llm_client.py",
      "MAX_RETRIES = 3",
      "MAX_RETRIES = 1      # P0 FIX: fail fast",
      "P0.1a MAX_RETRIES 3->1")

# P0.1b — Sniper threshold 0.97 -> 0.85
patch("src/pipeline/pipeline.py",
      ") >= 0.97",
      ") >= 0.85",
      "P0.1b threshold 0.97->0.85")

# P0.1c — import os in pipeline
patch("src/pipeline/pipeline.py",
      "import logging\nfrom typing import Any",
      "import logging\nimport os\nfrom typing import Any",
      "P0.1c import os")

# P0.1c — SNIPER_ONLY forced early-exit
SNIPER_OLD = """        if (
            getattr(
                state,
                "sniper_hit",
                False,
            )
            and getattr(
                state,
                "sniper_confidence",
                0.0,
            ) >= 0.85
        ):"""
SNIPER_NEW = """        _sniper_only = os.environ.get("SNIPER_ONLY") == "1"
        _sniper_hit = getattr(state, "sniper_hit", False)
        _sniper_conf = getattr(state, "sniper_confidence", 0.0)

        if _sniper_hit and (_sniper_only or _sniper_conf >= 0.85):"""
patch("src/pipeline/pipeline.py", SNIPER_OLD, SNIPER_NEW,
      "P0.1c SNIPER_ONLY early-exit")

# P0.2 — --sniper-only flag
ARG_OLD = """    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    args = parser.parse_args()"""
ARG_NEW = """    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--sniper-only",
        action="store_true",
        help="Skip all LLM pods; Sniper-only fast path",
    )

    args = parser.parse_args()

    if args.sniper_only:
        os.environ["SNIPER_ONLY"] = "1"
        os.environ["DISABLE_BGE"] = "1"
        print("[P0] SNIPER-ONLY mode: LLM + BGE disabled")"""
patch("eval/run_financebench.py", ARG_OLD, ARG_NEW, "P0.2 --sniper-only")

# ensure import os in eval
rf = ROOT / "eval/run_financebench.py"
if rf.exists():
    t = rf.read_text(encoding="utf-8")
    if "\nimport os" not in t and "import os\n" not in t:
        t = re.sub(r"(\nimport [a-zA-Z_]+\n)", r"\1import os\n", t, count=1)
        rf.write_text(t, encoding="utf-8")
        applied.append("P0.2 import os in eval")

print("="*60); print("PERIOD 0 PATCH REPORT"); print("="*60)
print("\nAPPLIED:");  [print("  OK", a) for a in applied]
print("\nSKIPPED:");  [print("  --", s) for s in skipped]
if failed:
    print("\nMANUAL CHECK:"); [print("  !!", f) for f in failed]
print("\nDone. Next: python eval/run_financebench.py --limit 2 --seed 42 --sniper-only")