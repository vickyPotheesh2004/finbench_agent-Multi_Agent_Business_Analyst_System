from __future__ import annotations

import re


class QueryType:

    NUMERIC = "numeric"
    RATIO = "ratio"
    NARRATIVE = "narrative"
    FORENSIC = "forensic"


def classify_query(query: str) -> str:

    q = query.lower()

    ratio_terms = [
        "margin",
        "ratio",
        "percentage",
        "growth",
        "operating margin",
    ]

    forensic_terms = [
        "fraud",
        "anomaly",
        "suspicious",
        "benford",
    ]

    numeric_terms = [
        "how much",
        "what was",
        "amount",
        "revenue",
        "income",
        "cash",
        "eps",
    ]

    if any(t in q for t in forensic_terms):
        return QueryType.FORENSIC

    if any(t in q for t in ratio_terms):
        return QueryType.RATIO

    if any(t in q for t in numeric_terms):
        return QueryType.NUMERIC

    return QueryType.NARRATIVE