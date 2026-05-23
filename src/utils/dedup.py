from __future__ import annotations


def deduplicate_chunks(chunks):

    seen = set()

    unique = []

    for chunk in chunks:

        text = (
            chunk.get("text", "")
            .strip()
        )

        key = hash(text)

        if key in seen:
            continue

        seen.add(key)

        unique.append(chunk)

    return unique