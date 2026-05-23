from __future__ import annotations

from src.utils.model_registry import get_tokenizer

MAX_PROMPT_TOKENS = 12000


def estimate_tokens(text: str) -> int:

    tokenizer = get_tokenizer()

    return len(
        tokenizer.encode(text)
    )


def trim_chunks_to_budget(chunks):

    kept = []

    total = 0

    for chunk in chunks:

        text = chunk.get("text", "")

        tokens = estimate_tokens(text)

        if total + tokens > MAX_PROMPT_TOKENS:
            break

        kept.append(chunk)

        total += tokens

    return kept