from __future__ import annotations

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

_BGE_MODEL = None
_TOKENIZER = None


def get_bge_model():

    global _BGE_MODEL

    if _BGE_MODEL is None:

        _BGE_MODEL = SentenceTransformer(
            "BAAI/bge-m3"
        )

    return _BGE_MODEL


def get_tokenizer():

    global _TOKENIZER

    if _TOKENIZER is None:

        _TOKENIZER = AutoTokenizer.from_pretrained(
            "BAAI/bge-m3"
        )

    return _TOKENIZER