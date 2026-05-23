"""
src/state/state.py

Central Pydantic State Object
FinBench Multi-Agent Business Analyst System
"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# ─────────────────────────────────────────────────────────────────────────────
# Valid Query Types
# ─────────────────────────────────────────────────────────────────────────────

QueryType = Literal[
    "numerical",
    "ratio",
    "multi_doc",
    "text",
    "forensic",
]

# ─────────────────────────────────────────────────────────────────────────────
# BAState
# ─────────────────────────────────────────────────────────────────────────────


class BAState(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        validate_assignment=False,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Session
    # ─────────────────────────────────────────────────────────────────────────

    session_id: str = "session"

    # ─────────────────────────────────────────────────────────────────────────
    # Document Metadata
    # ─────────────────────────────────────────────────────────────────────────

    document_path: str = ""

    company_name: str = "UNKNOWN"

    doc_type: str = "UNKNOWN"

    fiscal_year: str = "UNKNOWN"

    # ─────────────────────────────────────────────────────────────────────────
    # Raw Document
    # ─────────────────────────────────────────────────────────────────────────

    raw_text: str = ""

    cleaned_text: str = ""

    page_texts: List[str] = Field(
        default_factory=list
    )

    page_count: int = 0

    extracted_images: List[Dict] = Field(
        default_factory=list
    )

    table_cells: List[Dict] = Field(
        default_factory=list
    )

    section_tree: Dict[str, Any] = Field(
        default_factory=dict
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Chunking
    # ─────────────────────────────────────────────────────────────────────────

    chunk_count: int = 0

    bm25_index_path: str = ""

    chromadb_collection: str = ""

    chromadb_data_dir: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    query: str = ""

    query_type: QueryType = "text"

    query_complexity: str = "medium"

    difficulty_score: float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Routing
    # ─────────────────────────────────────────────────────────────────────────

    selected_pods: List[str] = Field(
        default_factory=list
    )

    winning_pod: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Sniper RAG
    # ─────────────────────────────────────────────────────────────────────────

    sniper_hit: bool = False

    sniper_confidence: float = 0.0

    sniper_answer: str = ""

    sniper_result: Optional[Any] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────────────────

    bm25_results: List[Dict] = Field(
        default_factory=list
    )

    bge_results: List[Dict] = Field(
        default_factory=list
    )

    retrieval_stage_1: List[Dict] = Field(
        default_factory=list
    )

    reranked_chunks: List[Dict] = Field(
        default_factory=list
    )

    retrieved_chunks: List[Dict] = Field(
        default_factory=list
    )

    retrieval_scores: Dict[str, float] = Field(
        default_factory=dict
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt Assembly
    # ─────────────────────────────────────────────────────────────────────────

    assembled_prompt: str = ""

    prompt_tokens: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Pod Outputs
    # ─────────────────────────────────────────────────────────────────────────

    analyst_answer: str = ""

    cfo_answer: str = ""

    auditor_answer: str = ""

    triguard_answer: str = ""

    mediator_answer: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence + Validation
    # ─────────────────────────────────────────────────────────────────────────

    confidence_score: float = 0.0

    validation_passed: bool = False

    validation_reason: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Final Output
    # ─────────────────────────────────────────────────────────────────────────

    final_answer: str = ""

    final_answer_pre_xgb: str = ""

    final_reasoning: str = ""

    citations: List[Dict] = Field(
        default_factory=list
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Explainability
    # ─────────────────────────────────────────────────────────────────────────

    shap_values: Dict[str, float] = Field(
        default_factory=dict
    )

    feature_importance: Dict[str, float] = Field(
        default_factory=dict
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ML Arbitration
    # ─────────────────────────────────────────────────────────────────────────

    xgb_score: float = 0.0

    xgb_prediction: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Output Files
    # ─────────────────────────────────────────────────────────────────────────

    output_docx_path: str = ""

    output_json_path: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Runtime
    # ─────────────────────────────────────────────────────────────────────────

    processing_time_seconds: float = 0.0

    error: str = ""

    warnings: List[str] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def add_warning(
        self,
        warning: str,
    ) -> None:

        if warning:

            self.warnings.append(
                str(warning)
            )

    def set_error(
        self,
        error: str,
    ) -> None:

        self.error = str(error)

    def clear_error(
        self,
    ) -> None:

        self.error = ""

    def reset_retrieval(
        self,
    ) -> None:

        self.bm25_results = []

        self.bge_results = []

        self.retrieval_stage_1 = []

        self.reranked_chunks = []

        self.retrieved_chunks = []

    def reset_answers(
        self,
    ) -> None:

        self.analyst_answer = ""

        self.cfo_answer = ""

        self.auditor_answer = ""

        self.triguard_answer = ""

        self.mediator_answer = ""

        self.final_answer = ""

        self.final_reasoning = ""

    def to_safe_dict(
        self,
    ) -> Dict[str, Any]:

        return self.model_dump(
            exclude={
                "raw_text",
                "cleaned_text",
            },
            exclude_none=True,
        )