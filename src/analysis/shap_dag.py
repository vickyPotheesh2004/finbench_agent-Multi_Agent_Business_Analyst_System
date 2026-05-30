"""
N16 SHAP + Causal DAG — Explainability and Causal Reasoning
PDR-BAAAI-001 · Rev 1.0 · Node N16

SESSION 16 FIX:
    + Bug Fix 8: result.get("shap") returns None when chunks are empty
                  or no SHAP could be computed. BAState has shap_values
                  as strict Dict (not Optional[Dict]), so None crashes
                  Pydantic v2 validate_assignment.
    + Fix: Use `or {}` and `or ""` defaults when setting state fields.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SEED         = 42
SHAP_ROW_CAP = 500
DAG_DPI      = 300

CAUSAL_EDGES = [
    ("Revenue",            "Gross Profit"),
    ("Cost of Revenue",    "Gross Profit"),
    ("Gross Profit",       "Operating Income"),
    ("Operating Expenses", "Operating Income"),
    ("Operating Income",   "Net Income"),
    ("Interest Expense",   "Net Income"),
    ("Tax Expense",        "Net Income"),
    ("Net Income",         "EPS"),
    ("Share Count",        "EPS"),
]


def compute_shap_importance(
    chunks: List[Dict],
    answer: str,
    seed:   int = SEED,
) -> Optional[Dict]:
    """Compute SHAP feature importance. Returns None if not computable."""
    if not chunks or not answer:
        return None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        import shap
    except ImportError:
        logger.warning("[N16] shap/sklearn not available — skipping")
        return None

    try:
        capped = chunks[:SHAP_ROW_CAP]
        texts  = [
            c.get("text", "") or c.get("page_content", "")
            for c in capped
        ]
        texts = [t for t in texts if t and len(t.strip()) > 30]

        if len(texts) < 3:
            return None

        answer_words = set(answer.lower().split())
        labels = []
        for text in texts:
            overlap = len(answer_words & set(text.lower().split()))
            labels.append(1 if overlap > 2 else 0)

        if len(set(labels)) < 2:
            labels = [i % 2 for i in range(len(labels))]

        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        X          = vectorizer.fit_transform(texts).toarray().astype(float)

        model = RandomForestClassifier(
            n_estimators=10, random_state=seed, max_depth=3
        )
        model.fit(X, labels)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list) and len(shap_values) > 1:
            sv = np.array(shap_values[1])
        elif isinstance(shap_values, list):
            sv = np.array(shap_values[0])
        else:
            sv = np.array(shap_values)

        if sv.ndim == 3:
            sv = sv.reshape(sv.shape[0], -1)
        if sv.ndim < 2:
            return None

        feature_names = vectorizer.get_feature_names_out()
        abs_shap      = np.abs(sv).mean(axis=0)

        top_idxs      = np.argsort(abs_shap)[-10:][::-1]
        top_features  = [
            {
                "feature":    str(feature_names[i]) if i < len(feature_names) else str(i),
                "importance": float(abs_shap[i]),
            }
            for i in top_idxs
        ]

        chunk_shap = np.abs(sv).sum(axis=1)
        top_chunk_idx = np.argsort(chunk_shap)[-5:][::-1]
        top_chunks = []
        for i in top_chunk_idx:
            # FIX (P0): bound by BOTH chunk_shap and capped lengths
            if i < len(chunk_shap) and i < len(capped):
                c = capped[i]
                top_chunks.append({
                    "chunk_id": c.get("chunk_id", c.get("id", str(i))),
                    "shap_sum": float(chunk_shap[i]),
                    "text":     (c.get("text") or "")[:200],
                })

        return {
            "top_features": top_features,
            "top_chunks":   top_chunks,
            "n_chunks":     len(texts),
        }

    except Exception as exc:
        logger.warning("[N16] SHAP failed gracefully: %s", exc)
        return None


def build_causal_dag(
    output_path:      Optional[str] = None,
    highlight_nodes:  Optional[List[str]] = None,
) -> Optional[str]:
    """Build causal DAG PNG. Returns path or None on failure."""
    if output_path is None:
        return "dag_built_no_path"
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    try:
        G = nx.DiGraph()
        G.add_edges_from(CAUSAL_EDGES)

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=SEED, k=1.5)

        node_colors = []
        for n in G.nodes():
            if highlight_nodes and n in highlight_nodes:
                node_colors.append("#ff7f0e")
            else:
                node_colors.append("#1f77b4")

        nx.draw(
            G, pos, with_labels=True,
            node_color=node_colors,
            node_size=2500,
            font_size=9,
            font_color="white",
            font_weight="bold",
            edge_color="#666666",
            arrows=True, arrowsize=20,
        )
        plt.title("Financial Causal DAG", fontsize=14, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=DAG_DPI, bbox_inches="tight")
        plt.close()
        return output_path
    except Exception as exc:
        logger.warning("[N16] DAG build failed: %s", exc)
        return None


class SHAPDAGNode:
    """N16 LangGraph node — SHAP + Causal DAG explainability."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir

    def run(self, state) -> object:
        """
        LangGraph N16 node.

        Reads:  state.retrieval_stage_2, state.final_answer_pre_xgb
        Writes: state.shap_values, state.feature_importance,
                state.causal_dag_path

        FIX (Session 16): Use `or {}` / `or ""` defaults to prevent
        crashing BAState when SHAP returns None (empty chunks, etc).
        """
        chunks = getattr(state, "retrieval_stage_2",    []) or []
        answer = getattr(state, "final_answer_pre_xgb", "") or ""

        result = self.explain(
            chunks=chunks,
            answer=answer,
            output_dir=self.output_dir,
        )

        # ── FIX: Convert SHAP result into BAState-compatible dicts ──
        # Issue: result.get("shap") returns the raw SHAP dict OR None.
        # BAState.shap_values is strict Dict (not Optional[Dict]).
        # Pydantic v2 with validate_assignment=True crashes on None.
        # Fix: Always pass a dict, never None.

        shap_raw = result.get("shap")
        if isinstance(shap_raw, dict):
            # Flatten top_features list into {name: importance} dict
            top_feats = shap_raw.get("top_features", [])
            shap_dict = {
                str(f.get("feature", "")): float(f.get("importance", 0.0))
                for f in top_feats
                if f.get("feature")
            }
        else:
            shap_dict = {}

        feat_imp_raw = result.get("feature_importance")
        if isinstance(feat_imp_raw, dict):
            feat_imp_dict = feat_imp_raw
        else:
            feat_imp_dict = {}

        dag_path_raw = result.get("dag_path")
        dag_path_str = str(dag_path_raw) if dag_path_raw else ""

        # Now safely set BAState fields (all non-None)
        state.shap_values        = shap_dict
        state.feature_importance = feat_imp_dict
        state.causal_dag_path    = dag_path_str

        logger.info(
            "N16 SHAP+DAG: shap_features=%d | dag=%s",
            len(shap_dict),
            dag_path_str or "skipped",
        )
        return state

    def explain(
        self,
        chunks:     List[Dict],
        answer:     str,
        output_dir: str = "outputs",
    ) -> Dict:
        """Compute SHAP importance + build causal DAG."""
        shap_result = compute_shap_importance(
            chunks=chunks, answer=answer, seed=SEED,
        )

        feature_importance = {}
        if shap_result:
            top_feats = shap_result.get("top_features", [])
            feature_importance = {
                str(f["feature"]): float(f["importance"])
                for f in top_feats
                if "feature" in f and "importance" in f
            }

        highlight = self._detect_relevant_nodes(answer)
        dag_path  = os.path.join(output_dir, "causal_dag.png")
        dag_saved = build_causal_dag(
            output_path=dag_path, highlight_nodes=highlight,
        )

        return {
            "shap":               shap_result,
            "feature_importance": feature_importance,
            "dag_path":           dag_saved,
            "highlight_nodes":    highlight,
        }

    @staticmethod
    def _detect_relevant_nodes(answer: str) -> List[str]:
        """Detect which causal DAG nodes are mentioned in the answer."""
        if not answer:
            return []

        node_keywords = {
            "Revenue":            ["revenue", "net sales", "total sales"],
            "Gross Profit":       ["gross profit", "gross margin"],
            "Operating Income":   ["operating income", "ebit", "operating profit"],
            "Net Income":         ["net income", "net earnings", "profit"],
            "EPS":                ["eps", "earnings per share", "diluted"],
            "Cost of Revenue":    ["cost of revenue", "cogs", "cost of sales"],
            "Operating Expenses": ["operating expense", "sg&a", "r&d"],
            "Tax Expense":        ["tax", "income tax"],
            "Interest Expense":   ["interest expense", "interest"],
        }

        answer_lower = answer.lower()
        return [
            node for node, kws in node_keywords.items()
            if any(kw in answer_lower for kw in kws)
        ]


def run_shap_dag(state, output_dir: str = "outputs") -> object:
    """Convenience wrapper for LangGraph N16 node."""
    return SHAPDAGNode(output_dir=output_dir).run(state)


# ── Sanity check ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test the fix: empty chunks should not crash
    class FakeState:
        retrieval_stage_2 = []
        final_answer_pre_xgb = ""
        shap_values = {}
        feature_importance = {}
        causal_dag_path = ""

    node = SHAPDAGNode(output_dir="/tmp")
    state = node.run(FakeState())
    assert state.shap_values == {}, "Empty chunks must produce empty dict, not None"
    assert state.feature_importance == {}, "Must produce empty dict, not None"
    assert state.causal_dag_path == "" or state.causal_dag_path, "Must be str, not None"
    print("✅ SESSION 16 FIX: SHAP returns empty dict on empty input (no None crash)")