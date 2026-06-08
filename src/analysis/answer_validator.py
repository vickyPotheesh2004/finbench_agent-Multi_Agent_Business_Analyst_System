from __future__ import annotations

import re


class AnswerValidator:
    """
    Lightweight final-answer sanity check.

    FIX (2026-06-07): the old `len(nums) > 20` rule silently rejected
    legitimate detailed answers (narrative driver lists, answers with
    citation blocks, multi-figure comparisons) as "number_spam" even
    when they were correct. Raised the ceiling to 60 and only trip it
    when the numbers also dominate the text (true gibberish), so real
    answers survive while genuine spam is still caught.
    """

    # Only flag as spam when there are MANY numbers AND they dominate.
    _MAX_NUMBERS = 60

    @staticmethod
    def validate(answer: str):

        if not answer:
            return False, "empty"

        if "RETRIEVAL_MISS" in answer:
            return False, "retrieval_miss"

        nums = re.findall(
            r"-?\d+\.?\d*",
            answer,
        )

        # FIX: only treat as spam when there are a LOT of numbers AND
        # they make up most of the tokens (i.e. it's gibberish, not a
        # legitimate answer that happens to cite several figures).
        if len(nums) > AnswerValidator._MAX_NUMBERS:
            tokens = answer.split()
            if tokens and (len(nums) / len(tokens)) > 0.6:
                return False, "number_spam"

        return True, "ok"
