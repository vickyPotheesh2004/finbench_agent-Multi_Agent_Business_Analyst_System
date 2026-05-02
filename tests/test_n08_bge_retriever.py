"""
tests/test_n08_bge_retriever.py
Tests for N08 BGE-M3 Semantic Retriever
PDR-BAAAI-001 · Rev 1.0

Gate M3 requirement: MRR@10 >= 0.85
Run: pytest tests/test_n08_bge_retriever.py -v
"""

import shutil
import time
from pathlib import Path

import pytest

from src.retrieval.bge_retriever import (
    BGERetriever,
    BGELangChainRetriever,
    run_bge,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    RETRIEVER_LABEL,
)
from src.state.ba_state import BAState


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "chunk_id":    "APPLE/10-K/FY2023/INCOME_STATEMENT/94/0",
        "text":        "Apple Inc total net sales were 383285 million dollars in fiscal year 2023. "
                       "Net income was 96995 million dollars. Diluted earnings per share were 6.13.",
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "section":     "INCOME_STATEMENT",
        "page":        94,
        "prefix":      "Apple Inc/10-K/FY2023/INCOME_STATEMENT/94",
    },
    {
        "chunk_id":    "APPLE/10-K/FY2023/BALANCE_SHEET/96/0",
        "text":        "Total assets were 352583 million dollars as of September 2023. "
                       "Total liabilities were 290437 million. Shareholders equity 62146 million.",
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "section":     "BALANCE_SHEET",
        "page":        96,
        "prefix":      "Apple Inc/10-K/FY2023/BALANCE_SHEET/96",
    },
    {
        "chunk_id":    "APPLE/10-K/FY2023/CASH_FLOW/98/0",
        "text":        "Net cash provided by operating activities was 114301 million in fiscal 2023. "
                       "Capital expenditures were 10959 million dollars.",
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "section":     "CASH_FLOW",
        "page":        98,
        "prefix":      "Apple Inc/10-K/FY2023/CASH_FLOW/98",
    },
    {
        "chunk_id":    "APPLE/10-K/FY2023/RISK_FACTORS/20/0",
        "text":        "Competition in each of the Company markets is intense. "
                       "The Company faces risks from global economic conditions.",
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "section":     "RISK_FACTORS",
        "page":        20,
        "prefix":      "Apple Inc/10-K/FY2023/RISK_FACTORS/20",
    },
    {
        "chunk_id":    "APPLE/10-K/FY2023/BUSINESS/5/0",
        "text":        "Apple Inc designs manufactures and markets smartphones computers and tablets. "
                       "iPhone represented 52 percent of total net revenue in fiscal year 2023.",
        "company":     "Apple Inc",
        "doc_type":    "10-K",
        "fiscal_year": "FY2023",
        "section":     "BUSINESS",
        "page":        5,
        "prefix":      "Apple Inc/10-K/FY2023/BUSINESS/5",
    },
]

COLLECTION_NAME = "test_bge_apple_fy2023"


@pytest.fixture(scope="module")
def built_collection(tmp_path_factory):
    """Build BGE collection once for all tests."""
    tmp_dir   = tmp_path_factory.mktemp("bge_test")
    retriever = BGERetriever(
        model_name = DEFAULT_MODEL,
        top_k      = DEFAULT_TOP_K,
        data_dir   = str(tmp_dir),
    )
    success = retriever.build_collection(
        chunks          = SAMPLE_CHUNKS,
        collection_name = COLLECTION_NAME,
        data_dir        = str(tmp_dir),
    )
    yield {
        "retriever":  retriever,
        "tmp_dir":    str(tmp_dir),
        "collection": COLLECTION_NAME,
        "success":    success,
    }
    # Cleanup
    try:
        import chromadb
        client = chromadb.PersistentClient(
            path=str(Path(tmp_dir) / "chromadb")
        )
        client.reset()
    except Exception:
        pass
    time.sleep(0.3)
    shutil.rmtree(str(tmp_dir), ignore_errors=True)


@pytest.fixture
def retriever():
    return BGERetriever(top_k=5)


# ── Group 1: Instantiation ────────────────────────────────────────────────────

class TestInstantiation:

    def test_01_instantiates(self, retriever):
        assert retriever is not None

    def test_02_default_model(self):
        r = BGERetriever()
        assert r.model_name == DEFAULT_MODEL

    def test_03_default_top_k(self):
        r = BGERetriever()
        assert r.top_k == DEFAULT_TOP_K

    def test_04_model_not_loaded_at_init(self, retriever):
        assert retriever._model is None

    def test_05_default_top_k_is_10(self):
        assert DEFAULT_TOP_K == 10

    def test_06_retriever_label(self):
        assert RETRIEVER_LABEL == "bge_m3"


# ── Group 2: Collection building ─────────────────────────────────────────────

class TestCollectionBuilding:

    def test_07_build_succeeds(self, built_collection):
        assert built_collection["success"] is True

    def test_08_collection_exists_after_build(self, built_collection):
        r = built_collection["retriever"]
        assert r.collection_exists(
            built_collection["collection"],
            built_collection["tmp_dir"],
        ) is True

    def test_09_collection_count_matches_chunks(self, built_collection):
        r     = built_collection["retriever"]
        count = r.get_collection_count(
            built_collection["collection"],
            built_collection["tmp_dir"],
        )
        assert count == len(SAMPLE_CHUNKS)

    def test_10_build_empty_chunks_returns_false(self, tmp_path):
        r = BGERetriever(data_dir=str(tmp_path))
        assert r.build_collection([], "empty_col", str(tmp_path)) is False

    def test_11_missing_collection_returns_none(self, built_collection):
        r = built_collection["retriever"]
        col = r._load_collection("nonexistent_col", built_collection["tmp_dir"])
        assert col is None


# ── Group 3: Retrieval results ────────────────────────────────────────────────

class TestRetrievalResults:

    def test_12_retrieve_returns_results(self, built_collection):
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "What was net income in 2023?",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
            top_k           = 3,
        )
        assert len(results) > 0

    def test_13_results_have_required_keys(self, built_collection):
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "total assets",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
        )
        required = {
            "text", "chunk_id", "rank", "bge_score",
            "retriever", "company", "doc_type",
            "fiscal_year", "section", "page", "prefix",
        }
        for res in results:
            assert required.issubset(res.keys()), \
                f"Missing keys: {required - res.keys()}"

    def test_14_retriever_label_is_bge_m3(self, built_collection):
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "net income",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
        )
        for res in results:
            assert res["retriever"] == RETRIEVER_LABEL

    def test_15_rank_starts_at_1(self, built_collection):
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "net income",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
        )
        assert results[0]["rank"] == 1

    def test_16_scores_between_0_and_1(self, built_collection):
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "earnings per share",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
        )
        for res in results:
            assert 0.0 <= res["bge_score"] <= 1.0

    def test_17_top_k_respected(self, built_collection):
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "net income",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
            top_k           = 2,
        )
        assert len(results) <= 2

    def test_18_empty_query_returns_empty(self, built_collection):
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
        )
        assert results == []

    def test_19_missing_collection_returns_empty(self, built_collection):
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "net income",
            collection_name = "does_not_exist",
            data_dir        = built_collection["tmp_dir"],
        )
        assert results == []

    def test_20_income_statement_chunk_ranks_high_for_revenue(
        self, built_collection
    ):
        """Revenue query should return income statement chunk in top-3."""
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "What was total net sales revenue?",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
            top_k           = 5,
        )
        sections = [res["section"] for res in results[:3]]
        assert "INCOME_STATEMENT" in sections, (
            f"INCOME_STATEMENT not in top-3 sections: {sections}"
        )

    def test_21_balance_sheet_chunk_ranks_high_for_assets(
        self, built_collection
    ):
        """Assets query should return balance sheet chunk in top-3."""
        r = built_collection["retriever"]
        results = r.retrieve(
            query           = "total assets and liabilities",
            collection_name = built_collection["collection"],
            data_dir        = built_collection["tmp_dir"],
            top_k           = 5,
        )
        sections = [res["section"] for res in results[:3]]
        assert "BALANCE_SHEET" in sections, (
            f"BALANCE_SHEET not in top-3 sections: {sections}"
        )


# ── Group 4: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_22_run_writes_retrieval_stage_1(self, built_collection):
        r     = built_collection["retriever"]
        state = BAState(
            session_id          = "t22",
            query               = "What was net income in FY2023?",
            chromadb_collection = built_collection["collection"],
        )
        r.data_dir = built_collection["tmp_dir"]
        state      = r.run(state)
        assert isinstance(state.retrieval_stage_1, list)
        assert len(state.retrieval_stage_1) > 0

    def test_23_seed_unchanged_after_run(self, built_collection):
        """C5: seed must remain 42"""
        r     = built_collection["retriever"]
        state = BAState(
            session_id          = "t23",
            query               = "net income",
            chromadb_collection = built_collection["collection"],
        )
        r.data_dir = built_collection["tmp_dir"]
        state      = r.run(state)
        assert state.seed == 42

    def test_24_empty_query_returns_empty_stage1(self, built_collection):
        r     = built_collection["retriever"]
        state = BAState(
            session_id          = "t24",
            query               = "",
            chromadb_collection = built_collection["collection"],
        )
        r.data_dir = built_collection["tmp_dir"]
        state      = r.run(state)
        assert state.retrieval_stage_1 == []

    def test_25_no_collection_returns_empty_stage1(self, built_collection):
        r     = built_collection["retriever"]
        state = BAState(
            session_id          = "t25",
            query               = "net income",
            chromadb_collection = "",
        )
        r.data_dir = built_collection["tmp_dir"]
        state      = r.run(state)
        assert state.retrieval_stage_1 == []

    def test_26_results_contain_c8_prefix(self, built_collection):
        """C8: results must carry company/doc_type/fiscal_year metadata"""
        r     = built_collection["retriever"]
        state = BAState(
            session_id          = "t26",
            query               = "net income 2023",
            chromadb_collection = built_collection["collection"],
        )
        r.data_dir = built_collection["tmp_dir"]
        state      = r.run(state)
        for res in state.retrieval_stage_1:
            assert res.get("company")     != ""
            assert res.get("doc_type")    != ""
            assert res.get("fiscal_year") != ""


# ── Group 5: LangChain integration ───────────────────────────────────────────

class TestLangChainIntegration:

    def test_27_as_langchain_retriever_not_none(self, built_collection):
        r  = built_collection["retriever"]
        lc = r.as_langchain_retriever(
            built_collection["collection"],
            built_collection["tmp_dir"],
        )
        assert lc is not None

    def test_28_langchain_invoke_returns_docs(self, built_collection):
        r  = built_collection["retriever"]
        lc = r.as_langchain_retriever(
            built_collection["collection"],
            built_collection["tmp_dir"],
        )
        docs = lc.invoke("net income 2023")
        assert len(docs) > 0

    def test_29_langchain_docs_have_page_content(self, built_collection):
        r  = built_collection["retriever"]
        lc = r.as_langchain_retriever(
            built_collection["collection"],
            built_collection["tmp_dir"],
        )
        docs = lc.invoke("total assets")
        for doc in docs:
            assert hasattr(doc, "page_content")
            assert len(doc.page_content) > 0

    def test_30_langchain_docs_have_metadata(self, built_collection):
        r  = built_collection["retriever"]
        lc = r.as_langchain_retriever(
            built_collection["collection"],
            built_collection["tmp_dir"],
        )
        docs = lc.invoke("net income")
        for doc in docs:
            assert hasattr(doc, "metadata")
            assert "chunk_id"  in doc.metadata
            assert "bge_score" in doc.metadata


# ── Group 6: Gate M3 — MRR@10 >= 0.85 ───────────────────────────────────────

class TestGateM3MRR:
    """
    Gate M3: MRR@10 >= 0.85 on the eval query set.

    MRR = Mean Reciprocal Rank.
    For each query, find the rank of the gold (correct) chunk.
    MRR = mean(1/rank) across all queries.

    Gold chunk mapping: query → expected section in top results.
    """

    GOLD_QUERIES = [
        ("What was total net sales revenue in fiscal 2023?",       "INCOME_STATEMENT"),
        ("What was net income in 2023?",                           "INCOME_STATEMENT"),
        ("What were diluted earnings per share?",                  "INCOME_STATEMENT"),
        ("What were total assets as of September 2023?",           "BALANCE_SHEET"),
        ("What were total liabilities?",                           "BALANCE_SHEET"),
        ("What was shareholders equity?",                          "BALANCE_SHEET"),
        ("What was operating cash flow in fiscal 2023?",           "CASH_FLOW"),
        ("What were capital expenditures?",                        "CASH_FLOW"),
        ("What are the main risk factors?",                        "RISK_FACTORS"),
        ("What products does Apple design and manufacture?",       "BUSINESS"),
    ]

    def _compute_mrr(self, built_collection) -> float:
        r            = built_collection["retriever"]
        reciprocal_ranks = []

        for query, gold_section in self.GOLD_QUERIES:
            results = r.retrieve(
                query           = query,
                collection_name = built_collection["collection"],
                data_dir        = built_collection["tmp_dir"],
                top_k           = 10,
            )
            rr = 0.0
            for res in results:
                if res["section"] == gold_section:
                    rr = 1.0 / res["rank"]
                    break
            reciprocal_ranks.append(rr)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    def test_31_mrr_at_10_above_085(self, built_collection):
        """Gate M3: MRR@10 must be >= 0.85"""
        mrr = self._compute_mrr(built_collection)
        assert mrr >= 0.85, (
            f"Gate M3 FAILED: MRR@10 = {mrr:.3f} < 0.85. "
            f"Domain fine-tuning required (Section 7.4). "
            f"Add 200 more hard negatives and run epoch 4."
        )

    def test_32_top1_accuracy_above_070(self, built_collection):
        """At least 70% of queries should have correct section at rank 1."""
        r      = built_collection["retriever"]
        hits   = 0
        for query, gold_section in self.GOLD_QUERIES:
            results = r.retrieve(
                query           = query,
                collection_name = built_collection["collection"],
                data_dir        = built_collection["tmp_dir"],
                top_k           = 1,
            )
            if results and results[0]["section"] == gold_section:
                hits += 1
        accuracy = hits / len(self.GOLD_QUERIES)
        assert accuracy >= 0.70, (
            f"Top-1 accuracy {accuracy:.1%} < 70% — "
            f"{hits}/{len(self.GOLD_QUERIES)} correct at rank 1"
        )

# ════════════════════════════════════════════════════════════════════════════
# BUG #8 — BGE-M3 should not load model when collection missing or disabled
# ════════════════════════════════════════════════════════════════════════════

class TestBug8LazyLoad:
    """Regression for Bug #8: model loaded even when not needed.

    Before fix: retrieve() with missing collection still loaded ~1.5GB model.
    After fix:  collection check happens first; model never loaded if not found.
    """

    def test_disabled_env_skips_model_load(self, tmp_path, monkeypatch):
        """DISABLE_BGE env var must prevent model load entirely."""
        monkeypatch.setenv("DISABLE_BGE", "1")
        r = BGERetriever(data_dir=str(tmp_path))
        results = r.retrieve(
            query="anything",
            collection_name="any",
            data_dir=str(tmp_path),
        )
        assert results == []
        # Critical: model must still be None after retrieve attempt
        assert r._model is None, (
            "Bug #8: DISABLE_BGE must prevent model load"
        )

    def test_disable_chromadb_also_disables_bge(self, tmp_path, monkeypatch):
        """DISABLE_CHROMADB env var must also disable BGE."""
        monkeypatch.setenv("DISABLE_CHROMADB", "1")
        r = BGERetriever(data_dir=str(tmp_path))
        # Both retrieve and build_collection should refuse
        assert r.retrieve("q", "c", str(tmp_path)) == []
        assert r.build_collection([{"text": "x"}], "c", str(tmp_path)) is False

    def test_missing_collection_skips_model_load(self, tmp_path):
        """Collection check must happen BEFORE model load."""
        r = BGERetriever(data_dir=str(tmp_path))
        results = r.retrieve(
            query="net income",
            collection_name="nonexistent_collection_xyz",
            data_dir=str(tmp_path),
        )
        assert results == []
        # Model must NOT have been loaded
        assert r._model is None, (
            "Bug #8: missing collection should skip model load "
            "(saved ~5s + 1.5GB RAM)"
        )

    def test_empty_query_skips_model_load(self, tmp_path):
        """Empty query must early-exit without model load."""
        r = BGERetriever(data_dir=str(tmp_path))
        r.retrieve(query="", collection_name="x", data_dir=str(tmp_path))
        assert r._model is None

    def test_empty_collection_name_skips_model_load(self, tmp_path):
        """Empty collection_name must early-exit without model load."""
        r = BGERetriever(data_dir=str(tmp_path))
        r.retrieve(query="x", collection_name="", data_dir=str(tmp_path))
        assert r._model is None