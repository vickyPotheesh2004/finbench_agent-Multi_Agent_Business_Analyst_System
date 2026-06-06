"""
tests/test_pipeline.py
Smoke tests for FinBenchPipeline import + instantiation.
Full integration test lives in test_integration.py (if present).

2026-06-05: removed stale references to _built / _ingestion_graph /
            _query_graph (these attributes were on an older pipeline
            class that has since been refactored away).
"""
import pytest


class TestPipelineImport:

    def test_01_pipeline_imports(self):
        from src.pipeline.pipeline import FinBenchPipeline
        assert FinBenchPipeline is not None

    def test_02_run_pipeline_imports(self):
        from src.pipeline.pipeline import run_pipeline
        assert run_pipeline is not None

    def test_03_pipeline_instantiates(self):
        from src.pipeline.pipeline import FinBenchPipeline
        p = FinBenchPipeline()
        assert p is not None

    def test_04_pipeline_has_required_attrs(self):
        """Current FinBenchPipeline must expose parallel + llm_client."""
        from src.pipeline.pipeline import FinBenchPipeline
        p = FinBenchPipeline()
        assert hasattr(p, "parallel")
        assert hasattr(p, "llm_client")

    def test_05_pipeline_has_ingest_and_query(self):
        """Pipeline must expose ingest() and query() methods."""
        from src.pipeline.pipeline import FinBenchPipeline
        p = FinBenchPipeline()
        assert callable(getattr(p, "ingest", None))
        assert callable(getattr(p, "query", None))
        assert callable(getattr(p, "run",   None))

    def test_06_safe_run_helper_exists(self):
        from src.pipeline.pipeline import _safe_run
        assert callable(_safe_run)

    def test_07_sniper_only_extract_fallback_exists(self):
        """FIX-B (2026-06-04) — the SNIPER_ONLY deterministic fallback."""
        from src.pipeline.pipeline import _sniper_only_extract_fallback
        assert callable(_sniper_only_extract_fallback)

    def test_08_query_type_constants_canonical(self):
        """Query type names must match BAState validator's canonical set."""
        from src.pipeline.pipeline import VALID_QUERY_TYPES
        assert VALID_QUERY_TYPES == {
            "numerical", "ratio", "multi_doc", "text", "forensic",
        }
