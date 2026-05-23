from __future__ import annotations

import re


IMPORTANT_TERMS = [
    "revenue",
    "net income",
    "operating income",
    "gross profit",
    "cash flow",
    "eps",
]


def normalize_query(query: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        query.lower(),
    ).strip()


def boost_financial_chunks(chunks):

    boosted = []

    for chunk in chunks:

        text = chunk.get("text", "").lower()

        score = float(
            chunk.get("score", 0.0)
        )

        for term in IMPORTANT_TERMS:

            if term in text:
                score += 0.05

        chunk["score"] = score

        boosted.append(chunk)

    return boosted