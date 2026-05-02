"""
N05 LR Difficulty Predictor — 3-Class Query Difficulty Classification
PDR-BAAAI-001 · Rev 1.0 · Node N05

Purpose:
    Classify every analyst query into easy / medium / hard AFTER N04 routing.
    Controls downstream pipeline behaviour:
        easy   → top-3 context, 1 PIV round, HITL threshold 0.65
        medium → top-3 context, 1 PIV round, HITL threshold 0.65
        hard   → top-5 context, 2 PIV rounds, HITL threshold 0.55

Architecture:
    Trained on 150 labelled questions (50 per class) using
    sklearn LogisticRegression on TF-IDF features.
    Model saved with joblib. Loads in <10ms at query time.

Constraints satisfied:
    C1  $0 cost — sklearn is free
    C2  100% local — zero network calls
    C5  seed=42 — random_state=42 on all sklearn objects
    C7  N/A — no LLM prompt at this node
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DIFFICULTY_CLASSES = ["easy", "medium", "hard"]

_MODEL_DIR           = os.path.join("models")
_MODEL_FILENAME      = "lr_difficulty.pkl"
_VECTORIZER_FILENAME = "lr_difficulty_vectorizer.pkl"
_SEED                = 42

# Difficulty configuration — controls PIV loop and retrieval behaviour
DIFFICULTY_CONFIG: Dict[str, Dict] = {
    "easy": {
        "context_window_size": 3,
        "piv_max_retries":     1,
        "hitl_threshold":      0.65,
        "description":         "Single fact lookup — direct answer expected",
    },
    "medium": {
        "context_window_size": 3,
        "piv_max_retries":     1,
        "hitl_threshold":      0.65,
        "description":         "Moderate reasoning — standard PIV loop",
    },
    "hard": {
        "context_window_size": 5,
        "piv_max_retries":     2,
        "hitl_threshold":      0.55,
        "description":         "Complex reasoning — wider context + extra retries",
    },
}


# ── Training data — 150 labelled questions (50 per class) ─────────────────────

TRAINING_DATA: List[Tuple[str, str]] = [
    # ── easy (50) — single fact, direct lookup ────────────────────────────────
    ("What was total net sales in FY2023?",                                "easy"),
    ("What was net income for the year?",                                  "easy"),
    ("What were total assets?",                                            "easy"),
    ("What was gross profit?",                                             "easy"),
    ("What was operating income?",                                         "easy"),
    ("What was cash at year end?",                                         "easy"),
    ("What was long-term debt?",                                           "easy"),
    ("What were capital expenditures?",                                    "easy"),
    ("What was R&D expense?",                                              "easy"),
    ("What was total revenue?",                                            "easy"),
    ("What was shareholders equity?",                                      "easy"),
    ("What were total liabilities?",                                       "easy"),
    ("What was basic EPS?",                                                "easy"),
    ("What was cost of revenue?",                                          "easy"),
    ("What was interest expense?",                                         "easy"),
    ("What was goodwill?",                                                 "easy"),
    ("What was deferred revenue?",                                         "easy"),
    ("What was accounts receivable?",                                      "easy"),
    ("What was inventory?",                                                "easy"),
    ("What were current assets?",                                          "easy"),
    ("What were current liabilities?",                                     "easy"),
    ("What was operating cash flow?",                                      "easy"),
    ("What was free cash flow?",                                           "easy"),
    ("What was the tax provision?",                                        "easy"),
    ("What were dividends paid?",                                          "easy"),
    ("What was net revenue?",                                              "easy"),
    ("What were total operating expenses?",                                "easy"),
    ("What was SG&A expense?",                                             "easy"),
    ("What was the book value per share?",                                 "easy"),
    ("What was amortization expense?",                                     "easy"),
    ("What was EBITDA?",                                                   "easy"),
    ("What was stock compensation expense?",                               "easy"),
    ("What was working capital?",                                          "easy"),
    ("What was the pension liability?",                                    "easy"),
    ("What were property plant and equipment?",                            "easy"),
    ("What was diluted EPS?",                                              "easy"),
    ("What was the effective tax rate?",                                   "easy"),
    ("What was the share repurchase amount?",                              "easy"),
    ("What were total current assets?",                                    "easy"),
    ("What were total current liabilities?",                               "easy"),
    ("What was net cash from operations?",                                 "easy"),
    ("What were capital expenditures in FY2022?",                         "easy"),
    ("What was gross margin in FY2023?",                                   "easy"),
    ("What was the revenue for the quarter?",                              "easy"),
    ("What was net income attributable to shareholders?",                  "easy"),
    ("What was the cash balance?",                                         "easy"),
    ("What were total borrowings?",                                        "easy"),
    ("What was the income tax expense?",                                   "easy"),
    ("What was depreciation expense?",                                     "easy"),
    ("What was accounts payable?",                                         "easy"),

    # ── medium (50) — requires formula or moderate context ───────────────────
    ("What was the gross margin percentage?",                              "medium"),
    ("Calculate the current ratio",                                        "medium"),
    ("What was the operating margin?",                                     "medium"),
    ("Calculate return on equity",                                         "medium"),
    ("What was the net profit margin?",                                    "medium"),
    ("What was return on assets?",                                         "medium"),
    ("Calculate the debt to equity ratio",                                 "medium"),
    ("What was the interest coverage ratio?",                              "medium"),
    ("What was the asset turnover ratio?",                                 "medium"),
    ("Calculate the quick ratio",                                          "medium"),
    ("What was the inventory turnover ratio?",                             "medium"),
    ("What was days sales outstanding?",                                   "medium"),
    ("What was the dividend payout ratio?",                                "medium"),
    ("Calculate the free cash flow yield",                                 "medium"),
    ("What was the EBITDA margin?",                                        "medium"),
    ("What was the revenue growth rate?",                                  "medium"),
    ("What are the main risk factors?",                                    "medium"),
    ("Describe the company's competitive advantages",                      "medium"),
    ("What did management say about guidance?",                            "medium"),
    ("Summarise the key points from MD&A",                                 "medium"),
    ("What is the company's strategy?",                                    "medium"),
    ("Describe the geographic revenue breakdown",                          "medium"),
    ("What are the key accounting policies?",                              "medium"),
    ("What litigation is disclosed?",                                      "medium"),
    ("What did management say about macroeconomic conditions?",            "medium"),
    ("Describe the company's capital allocation approach",                 "medium"),
    ("What are the covenant terms on debt?",                               "medium"),
    ("What cybersecurity risks are disclosed?",                            "medium"),
    ("What is the company's revenue recognition policy?",                  "medium"),
    ("Describe the executive compensation structure",                      "medium"),
    ("What was the capex to sales ratio?",                                 "medium"),
    ("Calculate the net debt to EBITDA ratio",                             "medium"),
    ("What was the return on capital employed?",                           "medium"),
    ("Calculate the cash conversion cycle",                                "medium"),
    ("What was the price to book ratio?",                                  "medium"),
    ("Calculate return on invested capital",                               "medium"),
    ("What was the financial leverage ratio?",                             "medium"),
    ("What was the SG&A as a percentage of revenue?",                      "medium"),
    ("What was the R&D intensity ratio?",                                   "medium"),
    ("Calculate the equity multiplier",                                    "medium"),
    ("What was the fixed asset turnover?",                                 "medium"),
    ("Calculate the altman z score",                                       "medium"),
    ("Summarise the auditor's report",                                     "medium"),
    ("What are the key assumptions in goodwill impairment?",               "medium"),
    ("Describe related party transactions",                                "medium"),
    ("What are the going concern disclosures?",                            "medium"),
    ("Describe the lease accounting judgements",                           "medium"),
    ("What are the commitments and contingencies?",                        "medium"),
    ("Summarise the pension benefit disclosures",                          "medium"),
    ("What are the key assumptions in stock option valuation?",            "medium"),

    # ── hard (50) — multi-step, cross-section, forensic, or multi-period ─────
    ("How did revenue compare between FY2022 and FY2023 and what drove the change?",  "hard"),
    ("Compare gross margins across the last three years and explain the trend",        "hard"),
    ("How has net income trended over five years and what caused fluctuations?",       "hard"),
    ("How did operating margins change from 2021 to 2023 and what drove the delta?",  "hard"),
    ("Compare the balance sheet in FY2022 versus FY2023 and identify key changes",    "hard"),
    ("Are there signs of earnings manipulation in the reported figures?",              "hard"),
    ("Does the Benford law test flag anomalies in the financial data?",                "hard"),
    ("Are accounts receivable growing faster than revenue?",                           "hard"),
    ("Does the cash flow statement contradict the income statement?",                  "hard"),
    ("Are there unusual related party transactions that could signal fraud?",          "hard"),
    ("Do the footnotes reveal off-balance-sheet arrangements?",                        "hard"),
    ("Are there signs of aggressive revenue recognition?",                             "hard"),
    ("Does the audit opinion contain qualifications or going concern language?",       "hard"),
    ("Are there material weaknesses in internal controls?",                            "hard"),
    ("Do accruals appear unusually large relative to revenue?",                        "hard"),
    ("Are there signs of cookie jar accounting reserves?",                             "hard"),
    ("Does the effective tax rate show unusual fluctuations across periods?",          "hard"),
    ("Are there unexplained changes in accounting policies?",                          "hard"),
    ("Does the Beneish M-score indicate earnings manipulation?",                       "hard"),
    ("Do the financial ratios show sudden unexplained improvements?",                  "hard"),
    ("Calculate the DuPont decomposition of ROE and explain each driver",              "hard"),
    ("Compare segment profitability across fiscal years and identify trends",          "hard"),
    ("How has the debt maturity profile changed and what are the refinancing risks?",  "hard"),
    ("Compare R&D spending as a percentage of revenue across three years",             "hard"),
    ("How has working capital evolved and what does it imply for liquidity?",          "hard"),
    ("Compare iPhone revenue to Services revenue growth over multiple periods",        "hard"),
    ("How did gross profit trend from 2019 to 2023 and what explains the change?",    "hard"),
    ("Compare the interest coverage ratio across years and assess solvency risk",      "hard"),
    ("How has stock compensation expense trended and what is the dilution impact?",    "hard"),
    ("Compare the book value per share across three years with ROE analysis",          "hard"),
    ("Are there signs of window dressing in the balance sheet at year end?",           "hard"),
    ("Does the management tone in MD&A contradict the financial numbers?",             "hard"),
    ("Are there signs of fictitious revenue in the receivables balances?",             "hard"),
    ("Do the gross margins show an unusual step change suggesting manipulation?",      "hard"),
    ("Does the cash conversion cycle show suspicious improvements?",                   "hard"),
    ("What is the free cash flow yield and how does it compare to peers?",             "hard"),
    ("Calculate the weighted average cost of capital and justify assumptions",         "hard"),
    ("How does the company's leverage compare to industry benchmarks?",                "hard"),
    ("What is the quality of earnings ratio and what does it imply?",                  "hard"),
    ("Analyse the sustainability of the dividend given cash flow trends",               "hard"),
    ("How did the Americas revenue compare to prior year across segments?",            "hard"),
    ("Compare net income margins between 2020 and 2023 with explanation",              "hard"),
    ("How has goodwill changed since the last acquisition and is impairment likely?",  "hard"),
    ("Compare deferred revenue balances across periods and assess recognition risk",   "hard"),
    ("How did the effective tax rate change year over year and what drove it?",        "hard"),
    ("Compare the pension obligations across years and assess funding adequacy",       "hard"),
    ("Does inventory growth outpace COGS and does it suggest channel stuffing?",       "hard"),
    ("Are there discrepancies between reported and pro forma earnings?",               "hard"),
    ("Do the expense trends show signs of undisclosed cost cuts affecting quality?",   "hard"),
    ("Are there anomalies in the geographic revenue breakdown across periods?",        "hard"),
]


# ── LRDifficultyPredictor ─────────────────────────────────────────────────────

class LRDifficultyPredictor:
    """
    N05 LR Difficulty Predictor.

    Classifies analyst queries into easy / medium / hard using
    LogisticRegression on TF-IDF features.

    Two usage modes:
        1. predictor.predict(query) → (difficulty, confidence, config)
        2. predictor.run(ba_state)  → BAState (LangGraph pipeline node)

    Training:
        predictor.train()       — trains on 150 labelled questions
        predictor.save()        — saves model + vectorizer with joblib
        predictor.load()        — loads from disk

    Auto-trains on first use if no saved model exists.
    """

    def __init__(self, model_dir: str = _MODEL_DIR) -> None:
        self.model_dir   = model_dir
        self._model      = None
        self._vectorizer = None
        self._is_trained = False

    # ── LangGraph pipeline node entry point ───────────────────────────────────

    def run(self, state) -> object:
        """
        LangGraph N05 node entry point. Fires after N04 CART Router.

        Reads:  state.query
        Writes: state.query_difficulty, state.context_window_size
                (may override N04 context_window_size for hard queries)

        Args:
            state: BAState object

        Returns:
            BAState with difficulty fields populated
        """
        query = getattr(state, "query", "") or ""

        if not query:
            logger.warning("N05 LR: empty query — defaulting to 'medium'")
            state.query_difficulty    = "medium"
            return state

        difficulty, confidence, config = self.predict(query)

        state.query_difficulty = difficulty

        # Hard queries override context window size from N04
        if difficulty == "hard":
            state.context_window_size = config["context_window_size"]

        logger.info(
            "N05 LR: difficulty=%s | confidence=%.3f | "
            "context_window=%d | piv_retries=%d",
            difficulty, confidence,
            config["context_window_size"],
            config["piv_max_retries"],
        )
        return state

    # ── Core prediction method ────────────────────────────────────────────────

    def predict(self, query: str) -> Tuple[str, float, Dict]:
        """
        Predict difficulty of a query.

        Args:
            query: Analyst question string

        Returns:
            Tuple of:
                difficulty : one of easy/medium/hard
                confidence : float 0.0–1.0 from predict_proba
                config     : difficulty config dict
        """
        self._ensure_trained()

        features   = self._vectorizer.transform([query])
        prediction = self._model.predict(features)[0]
        proba      = self._model.predict_proba(features)[0]
        confidence = float(max(proba))

        return prediction, confidence, DIFFICULTY_CONFIG[prediction]

    def predict_batch(
        self, queries: List[str]
    ) -> List[Tuple[str, float, Dict]]:
        """Predict difficulty for multiple queries efficiently."""
        self._ensure_trained()
        features    = self._vectorizer.transform(queries)
        predictions = self._model.predict(features)
        probas      = self._model.predict_proba(features)
        return [
            (pred, float(max(prob)), DIFFICULTY_CONFIG[pred])
            for pred, prob in zip(predictions, probas)
        ]

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        training_data: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict:
        """
        Train LogisticRegression classifier on labelled questions.

        Args:
            training_data: List of (query, label) tuples.
                           Defaults to built-in 150-question set.

        Returns:
            Dict with train_accuracy and n_samples.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        data   = training_data or TRAINING_DATA
        texts  = [q for q, _ in data]
        labels = [l for _, l in data]

        # TF-IDF vectoriser
        self._vectorizer = TfidfVectorizer(
            ngram_range  = (1, 3),
            max_features = 1500,
            sublinear_tf = True,
        )
        features = self._vectorizer.fit_transform(texts)

        # Logistic Regression — C5 seed=42
        self._model = LogisticRegression(
            max_iter     = 1000,
            random_state = _SEED,
            C            = 1.0,
            solver       = "lbfgs",
        )
        self._model.fit(features, labels)

        preds          = self._model.predict(features)
        train_accuracy = sum(p == l for p, l in zip(preds, labels)) / len(labels)

        self._is_trained = True

        logger.info(
            "N05 LR trained: %d samples | train_accuracy=%.3f",
            len(data), train_accuracy,
        )
        return {
            "train_accuracy": train_accuracy,
            "n_samples":      len(data),
            "n_classes":      len(DIFFICULTY_CLASSES),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, model_dir: Optional[str] = None) -> str:
        """Save trained model and vectorizer to disk."""
        import joblib

        save_dir        = model_dir or self.model_dir
        os.makedirs(save_dir, exist_ok=True)
        model_path      = os.path.join(save_dir, _MODEL_FILENAME)
        vectorizer_path = os.path.join(save_dir, _VECTORIZER_FILENAME)

        joblib.dump(self._model,      model_path)
        joblib.dump(self._vectorizer, vectorizer_path)

        logger.info("N05 LR saved: %s", model_path)
        return model_path

    def load(self, model_dir: Optional[str] = None) -> bool:
        """Load model and vectorizer from disk."""
        import joblib

        load_dir        = model_dir or self.model_dir
        model_path      = os.path.join(load_dir, _MODEL_FILENAME)
        vectorizer_path = os.path.join(load_dir, _VECTORIZER_FILENAME)

        if not os.path.exists(model_path):
            return False
        if not os.path.exists(vectorizer_path):
            return False

        try:
            self._model      = joblib.load(model_path)
            self._vectorizer = joblib.load(vectorizer_path)
            self._is_trained = True
            logger.info("N05 LR loaded from: %s", model_path)
            return True
        except Exception as exc:
            logger.warning("N05 LR load failed: %s", exc)
            return False

    def is_trained(self) -> bool:
        return self._is_trained

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_trained(self) -> None:
        """Auto-train if model not yet trained or loaded.

        Bug #7 fix: persist trained model to disk so next instance loads
        instead of retraining (saved ~3s per pipeline run).
        """
        if not self._is_trained:
            loaded = self.load()
            if not loaded:
                logger.info("N05 LR: no saved model — training now")
                self.train()
                # Bug #7: persist immediately
                try:
                    self.save()
                    logger.info("N05 LR: model saved for reuse")
                except Exception as exc:
                    logger.warning(
                        "N05 LR: save after train failed: %s", exc
                    )


# ── Convenience wrapper for LangGraph N05 node ───────────────────────────────

def run_lr_difficulty(state, model_dir: str = _MODEL_DIR) -> object:
    """
    Convenience wrapper used by the LangGraph pipeline node N05.
    """
    predictor = LRDifficultyPredictor(model_dir=model_dir)
    return predictor.run(state)