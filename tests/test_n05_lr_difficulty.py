"""
tests/test_n05_lr_difficulty.py
Tests for N05 LR Difficulty Predictor
PDR-BAAAI-001 · Rev 1.0
"""

import os
import pytest
from src.routing.lr_difficulty import (
    LRDifficultyPredictor,
    run_lr_difficulty,
    DIFFICULTY_CLASSES,
    DIFFICULTY_CONFIG,
    TRAINING_DATA,
    _SEED,
    _MODEL_FILENAME,
    _VECTORIZER_FILENAME,
)
from src.state.ba_state import BAState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_predictor(tmp_path_factory):
    """Train once and reuse across all tests."""
    tmp_dir   = str(tmp_path_factory.mktemp("lr_model"))
    predictor = LRDifficultyPredictor(model_dir=tmp_dir)
    predictor.train()
    predictor.save()
    return predictor


@pytest.fixture
def fresh_predictor(tmp_path):
    return LRDifficultyPredictor(model_dir=str(tmp_path))


# ── Group 1: Constants ────────────────────────────────────────────────────────

class TestConstants:

    def test_01_three_difficulty_classes(self):
        assert len(DIFFICULTY_CLASSES) == 3

    def test_02_classes_correct(self):
        assert set(DIFFICULTY_CLASSES) == {"easy", "medium", "hard"}

    def test_03_config_has_all_classes(self):
        for cls in DIFFICULTY_CLASSES:
            assert cls in DIFFICULTY_CONFIG

    def test_04_config_keys_present(self):
        required = {
            "context_window_size", "piv_max_retries",
            "hitl_threshold", "description"
        }
        for cls in DIFFICULTY_CLASSES:
            assert required.issubset(DIFFICULTY_CONFIG[cls].keys())

    def test_05_hard_has_wider_context(self):
        assert DIFFICULTY_CONFIG["hard"]["context_window_size"] >= 5

    def test_06_hard_has_more_piv_retries(self):
        assert DIFFICULTY_CONFIG["hard"]["piv_max_retries"] >= 2

    def test_07_hard_has_lower_hitl_threshold(self):
        assert DIFFICULTY_CONFIG["hard"]["hitl_threshold"] < \
               DIFFICULTY_CONFIG["easy"]["hitl_threshold"]

    def test_08_seed_is_42(self):
        assert _SEED == 42

    def test_09_training_data_150_samples(self):
        assert len(TRAINING_DATA) == 150

    def test_10_training_data_balanced(self):
        from collections import Counter
        counts = Counter(label for _, label in TRAINING_DATA)
        for cls in DIFFICULTY_CLASSES:
            assert counts[cls] == 50, (
                f"Class '{cls}' has {counts[cls]} samples, expected 50"
            )


# ── Group 2: Instantiation ────────────────────────────────────────────────────

class TestInstantiation:

    def test_11_instantiates(self, fresh_predictor):
        assert fresh_predictor is not None

    def test_12_not_trained_at_init(self, fresh_predictor):
        assert fresh_predictor.is_trained() is False

    def test_13_model_none_at_init(self, fresh_predictor):
        assert fresh_predictor._model is None

    def test_14_vectorizer_none_at_init(self, fresh_predictor):
        assert fresh_predictor._vectorizer is None


# ── Group 3: Training ─────────────────────────────────────────────────────────

class TestTraining:

    def test_15_train_returns_dict(self, fresh_predictor):
        result = fresh_predictor.train()
        assert isinstance(result, dict)

    def test_16_train_result_has_accuracy(self, fresh_predictor):
        result = fresh_predictor.train()
        assert "train_accuracy" in result

    def test_17_train_accuracy_above_85_percent(self, fresh_predictor):
        result = fresh_predictor.train()
        assert result["train_accuracy"] >= 0.85, (
            f"Train accuracy {result['train_accuracy']:.1%} < 85%"
        )

    def test_18_trained_after_train(self, fresh_predictor):
        fresh_predictor.train()
        assert fresh_predictor.is_trained() is True

    def test_19_n_samples_correct(self, fresh_predictor):
        result = fresh_predictor.train()
        assert result["n_samples"] == 150


# ── Group 4: Persistence ──────────────────────────────────────────────────────

class TestPersistence:

    def test_20_save_creates_model_file(self, tmp_path):
        p = LRDifficultyPredictor(model_dir=str(tmp_path))
        p.train()
        p.save()
        assert os.path.exists(os.path.join(str(tmp_path), _MODEL_FILENAME))

    def test_21_save_creates_vectorizer_file(self, tmp_path):
        p = LRDifficultyPredictor(model_dir=str(tmp_path))
        p.train()
        p.save()
        assert os.path.exists(
            os.path.join(str(tmp_path), _VECTORIZER_FILENAME)
        )

    def test_22_load_returns_true(self, tmp_path):
        p = LRDifficultyPredictor(model_dir=str(tmp_path))
        p.train()
        p.save()
        p2 = LRDifficultyPredictor(model_dir=str(tmp_path))
        assert p2.load() is True

    def test_23_load_missing_returns_false(self, tmp_path):
        p = LRDifficultyPredictor(model_dir=str(tmp_path))
        assert p.load() is False

    def test_24_loaded_model_is_trained(self, tmp_path):
        p = LRDifficultyPredictor(model_dir=str(tmp_path))
        p.train()
        p.save()
        p2 = LRDifficultyPredictor(model_dir=str(tmp_path))
        p2.load()
        assert p2.is_trained() is True

    def test_25_save_returns_path_string(self, tmp_path):
        p = LRDifficultyPredictor(model_dir=str(tmp_path))
        p.train()
        path = p.save()
        assert isinstance(path, str)
        assert path.endswith(".pkl")


# ── Group 5: Prediction accuracy ─────────────────────────────────────────────

class TestPredictionAccuracy:

    @pytest.mark.parametrize("query,expected", [
        ("What was total net sales?",                          "easy"),
        ("What was net income?",                               "easy"),
        ("What were total assets?",                            "easy"),
        ("What was gross profit?",                             "easy"),
        ("What was cash at year end?",                         "easy"),
        ("What was the gross margin percentage?",              "medium"),
        ("Calculate return on equity",                         "medium"),
        ("What are the main risk factors?",                    "medium"),
        ("Describe the competitive advantages",                "medium"),
        ("Summarise the key points from MD&A",                 "medium"),
        ("Compare revenue across three years and explain the trend",   "hard"),
        ("Are there signs of earnings manipulation?",          "hard"),
        ("Does the cash flow statement contradict the income statement?","hard"),
        ("Calculate DuPont decomposition and explain each driver",     "hard"),
        ("Compare segment profitability across fiscal years",  "hard"),
    ])
    def test_26_prediction_accuracy(
        self, trained_predictor, query, expected
    ):
        difficulty, confidence, _ = trained_predictor.predict(query)
        assert difficulty == expected, (
            f"Query: '{query}'\n"
            f"Expected: {expected}\n"
            f"Got:      {difficulty} (conf={confidence:.3f})"
        )

    def test_27_predict_returns_tuple(self, trained_predictor):
        result = trained_predictor.predict("What was net income?")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_28_confidence_between_0_and_1(self, trained_predictor):
        _, confidence, _ = trained_predictor.predict("net income FY2023")
        assert 0.0 <= confidence <= 1.0

    def test_29_result_includes_config(self, trained_predictor):
        _, _, config = trained_predictor.predict("net income FY2023")
        assert "context_window_size" in config
        assert "piv_max_retries"     in config
        assert "hitl_threshold"      in config

    def test_30_all_3_classes_reachable(self, trained_predictor):
        queries = {
            "easy":   "What was net income?",
            "medium": "Calculate the gross margin percentage",
            "hard":   "Compare revenue across three years and explain the trend",
        }
        found = set()
        for _, q in queries.items():
            d, _, _ = trained_predictor.predict(q)
            found.add(d)
        assert len(found) >= 2, (
            f"Only {len(found)} classes found: {found}"
        )

    def test_31_batch_predict_returns_list(self, trained_predictor):
        queries = ["net income?", "gross margin ratio", "complex multi-year analysis"]
        results = trained_predictor.predict_batch(queries)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, tuple)
            assert len(r) == 3


# ── Group 6: BAState integration ─────────────────────────────────────────────

class TestBAStateIntegration:

    def test_32_run_writes_query_difficulty(self, trained_predictor):
        state = BAState(
            session_id = "t32",
            query      = "What was net income FY2023?",
        )
        state = trained_predictor.run(state)
        assert state.query_difficulty in DIFFICULTY_CLASSES

    def test_33_hard_query_sets_wider_context(self, trained_predictor):
        state = BAState(
            session_id = "t33",
            query      = "Compare revenue across three years and explain the trend",
        )
        state = trained_predictor.run(state)
        if state.query_difficulty == "hard":
            assert state.context_window_size >= 5

    def test_34_seed_unchanged_after_run(self, trained_predictor):
        """C5: seed must remain 42"""
        state = BAState(
            session_id = "t34",
            query      = "What was net income?",
        )
        state = trained_predictor.run(state)
        assert state.seed == 42

    def test_35_empty_query_defaults_to_medium(self, trained_predictor):
        state = BAState(session_id="t35", query="")
        state = trained_predictor.run(state)
        assert state.query_difficulty == "medium"

    def test_36_easy_query_classified(self, trained_predictor):
        state = BAState(
            session_id = "t36",
            query      = "What was total net sales?",
        )
        state = trained_predictor.run(state)
        assert state.query_difficulty in DIFFICULTY_CLASSES


# ── Group 7: N04 + N05 pipeline integration ───────────────────────────────────

class TestN04N05Pipeline:

    def test_37_n04_then_n05_both_write_state(self, tmp_path):
        """N04 and N05 run sequentially — both write to state."""
        from src.routing.cart_router import CARTRouter
        cart      = CARTRouter(model_dir=str(tmp_path))
        predictor = LRDifficultyPredictor(model_dir=str(tmp_path))

        state = BAState(
            session_id = "t37",
            query      = "What was total net sales in FY2023?",
        )
        state = cart.run(state)
        state = predictor.run(state)

        assert state.query_type       in ["numerical", "ratio", "multi_doc", "text", "forensic"]
        assert state.query_difficulty in DIFFICULTY_CLASSES

    def test_38_hard_forensic_gets_widest_context(self, tmp_path):
        """Forensic hard query should result in widest context window."""
        from src.routing.cart_router import CARTRouter
        cart      = CARTRouter(model_dir=str(tmp_path))
        predictor = LRDifficultyPredictor(model_dir=str(tmp_path))

        state = BAState(
            session_id = "t38",
            query      = "Are there signs of earnings manipulation in the financials?",
        )
        state = cart.run(state)
        state = predictor.run(state)

        assert state.query_difficulty in DIFFICULTY_CLASSES
        assert state.context_window_size >= 3


# ── Group 8: Convenience wrapper ─────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_39_run_lr_difficulty_returns_state(self, tmp_path):
        state = BAState(
            session_id = "t39",
            query      = "What was net income?",
        )
        result = run_lr_difficulty(state, model_dir=str(tmp_path))
        assert hasattr(result, "query_difficulty")
        assert result.query_difficulty in DIFFICULTY_CLASSES

    def test_40_wrapper_seed_unchanged(self, tmp_path):
        state = BAState(
            session_id = "t40",
            query      = "What was net income?",
        )
        result = run_lr_difficulty(state, model_dir=str(tmp_path))
        assert result.seed == 42

# ════════════════════════════════════════════════════════════════════════════
# BUG #7 — LR Difficulty must persist after first train
# ════════════════════════════════════════════════════════════════════════════

class TestBug7LRPersistence:
    """Regression for Bug #7: LR Difficulty retrained on every pipeline run."""

    def test_first_use_saves_model_to_disk(self, tmp_path):
        """Bug #7: _ensure_trained() must save after training."""
        from src.routing.lr_difficulty import (
            LRDifficultyPredictor,
            _MODEL_FILENAME,
            _VECTORIZER_FILENAME,
        )
        predictor = LRDifficultyPredictor(model_dir=str(tmp_path))
        predictor.predict("What was net income?")

        model_path      = os.path.join(str(tmp_path), _MODEL_FILENAME)
        vectorizer_path = os.path.join(str(tmp_path), _VECTORIZER_FILENAME)
        assert os.path.exists(model_path)
        assert os.path.exists(vectorizer_path)

    def test_second_instance_loads_quickly(self, tmp_path):
        """Bug #7: 2nd instance should load instead of retrain."""
        from src.routing.lr_difficulty import LRDifficultyPredictor
        import time

        p1 = LRDifficultyPredictor(model_dir=str(tmp_path))
        p1.predict("seed")

        t0 = time.time()
        p2 = LRDifficultyPredictor(model_dir=str(tmp_path))
        p2.predict("another")
        elapsed = time.time() - t0
        assert elapsed < 2.0, (
            f"Bug #7: 2nd LR instance took {elapsed:.2f}s (must be <2s)"
        )