"""
src/utils/model_registry.py
DEPRECATED — DO NOT USE.

This file used to load BAAI/bge-m3 globally, but bge_retriever.py loads its
own model (BAAI/bge-small-en-v1.5) directly. The two model references were
inconsistent and this file was never imported anywhere in the codebase.

Kept as an empty stub so `import src.utils.model_registry` never fails for
anyone with stale references. The functions are intentionally absent.

If you need centralised model loading, see:
  - src/retrieval/bge_retriever.py   (BGE for retrieval)
  - src/retrieval/rrf_reranker.py    (cross-encoder reranker)
  - src/analysis/piv_loop.py         (Ollama Llama 3.1 8B for LLM)
"""
from __future__ import annotations

# Intentionally empty. See module docstring.
