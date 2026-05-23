from __future__ import annotations


def filter_by_fiscal_year(
    chunks,
    fiscal_year: str,
):

    if not fiscal_year:
        return chunks

    filtered = []

    fy = fiscal_year.lower()

    for chunk in chunks:

        text = chunk.get(
            "text",
            "",
        ).lower()

        if fy in text:
            filtered.append(chunk)

    return filtered or chunks