from __future__ import annotations

import re


class AnswerValidator:

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

        if len(nums) > 20:
            return False, "number_spam"

        return True, "ok"