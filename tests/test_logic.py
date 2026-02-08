"""
Comprehensive test suite for the Causal Uplift Optimizer.

Covers: data_loader, model (T/S/X-Learner), optimizer.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path when running this file directly
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    _coerce_numeric_columns,
    _generate_synthetic_hillstrom,
    _one_hot_encode,
    _sanitize_column_names,
    generic_preprocess,
    preprocess,
)
from src.model import BaseUpliftModel, SLearner, TLearner, XLearner
from src.optimizer import OptimizationResult, optimize_budget

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return a small reproducible (X, treatment, outcome) tuple."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    treatment = pd.Series(rng.integers(0, 2, n), name="treatment")
    outcome = pd.Series(rng.integers(0, 2, n), name="outcome")
    return X, treatment, outcome


@pytest.fixture
def scored_df() -> pd.DataFrame:
    """DataFrame with pre-computed uplift scores for optimizer tests."""
    return pd.DataFrame(
        {
            "uplift": [0.9, 0.5, 0.2, -0.1],
            "outcome": [1, 0, 1, 0],
            "treatment": [1, 1, 0, 0],
        }
    )


# ──────────────────────────────────────────────
# data_loader — helpers
# ──────────────────────────────────────────────

class TestCoerceNumericColumns:
    """Tests for _coerce_numeric_columns."""

    def test_cleans_bracket_wrapped_numbers(self):
        df = pd.DataFrame({"a": ["[1.5]", "[2.0]", "[3.3]"]})
        result = _coerce_numeric_columns(df)
        assert result["a"].dtype == np.float64
        assert result["a"].tolist() == pytest.approx([1.5, 2.0, 3.3])

    def test_cleans_quote_wrapped_numbers(self):
        df = pd.DataFrame({"a": ["'10'", "'20'", "'30'"]})
        result = _coerce_numeric_columns(df)
        assert pd.api.types.is_numeric_dtype(result["a"])

    def test_preserves_true_categoricals(self):
        df = pd.DataFrame({"city": ["Paris", "London", "Berlin"]})
        result = _coerce_numeric_columns(df)
        assert result["city"].dtype == object

    def test_mixed_column_below_threshold_stays_object(self):
        # Only 1 out of 4 is numeric → below 50 % threshold
        df = pd.DataFrame({"a": ["hello", "world", "foo", "42"]})
        result = _coerce_numeric_columns(df)
        assert result["a"].dtype == object

    def test_already_numeric_columns_untouched(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = _coerce_numeric_columns(df)
        pd.testing.assert_frame_equal(result, df)


class TestSanitizeColumnNames:
    def test_removes_brackets_and_angles(self):
        df = pd.DataFrame({"col[0]": [1], "col<1>": [2], "normal": [3]})
        result = _sanitize_column_names(df)
        assert list(result.columns) == ["col0", "col1", "normal"]


class TestOneHotEncode:
    def test_encodes_specified_columns(self):
        df = pd.DataFrame({"color": ["red", "blue", "red"], "val": [1, 2, 3]})
        result = _one_hot_encode(df, columns=["color"], drop_first=True)
        assert "color" not in result.columns
        assert "val" in result.columns

    def test_empty_columns_list_returns_copy(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = _one_hot_encode(df, columns=[])
        pd.testing.assert_frame_equal(result, df)


# ──────────────────────────────────────────────
# data_loader — public API
# ──────────────────────────────────────────────

class TestPreprocess:
    def test_hillstrom_shapes(self):
        raw = _generate_synthetic_hillstrom(n=500, seed=1)
        X, t, y = preprocess(raw)

        assert len(X) == 500
        assert len(t) == 500
        assert len(y) == 500
        assert t.isin([0, 1]).all()
        assert y.isin([0, 1]).all()

    def test_features_are_numeric(self):
        raw = _generate_synthetic_hillstrom(n=100, seed=2)
        X, _, _ = preprocess(raw)
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col]), f"{col} is not numeric"


class TestGenericPreprocess:
    def test_basic_preprocessing(self):
        df = pd.DataFrame(
            {"treat": [0, 1, 0, 1], "out": [1, 0, 1, 0], "f1": [10, 20, 30, 40]}
        )
        X, t, y = generic_preprocess(df, "treat", "out")
        assert list(X.columns) == ["f1"]
        assert len(X) == 4

    def test_drops_nan_rows(self):
        df = pd.DataFrame(
            {"treat": [0, 1, None], "out": [1, 0, 1], "f1": [10, 20, 30]}
        )
        X, t, y = generic_preprocess(df, "treat", "out")
        assert len(X) == 2

    def test_raises_on_empty_after_dropna(self):
        df = pd.DataFrame(
            {"treat": [None, None], "out": [None, None], "f1": [None, None]}
        )
        with pytest.raises(ValueError, match="empty"):
            generic_preprocess(df, "treat", "out")

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generic_preprocess(df, "treat", "out")

    def test_dirty_bracket_data_is_cleaned(self):
        df = pd.DataFrame(
            {
                "treat": [0, 1, 0, 1],
                "out": [0, 1, 0, 1],
                "feat": ["[1.5]", "[2.0]", "[3.3]", "[4.1]"],
            }
        )
        X, _, _ = generic_preprocess(df, "treat", "out")
        assert X["feat"].dtype == np.float64

    def test_xgboost_safe_column_names(self):
        df = pd.DataFrame(
            {
                "treat": [0, 1],
                "out": [0, 1],
                "cat": ["a[1]", "b<2>"],
            }
        )
        X, _, _ = generic_preprocess(df, "treat", "out")
        for col in X.columns:
            assert "[" not in col and "]" not in col and "<" not in col


# ──────────────────────────────────────────────
# model — all learners
# ──────────────────────────────────────────────

class TestModels:
    @pytest.mark.parametrize("ModelClass", [TLearner, SLearner, XLearner])
    def test_train_and_predict(self, ModelClass, dummy_data):
        X, t, y = dummy_data
        model = ModelClass()
        model.train(X, t, y)
        uplift = model.predict_uplift(X)

        assert isinstance(uplift, np.ndarray)
        assert len(uplift) == len(X)
        assert not np.isnan(uplift).any()

    @pytest.mark.parametrize("ModelClass", [TLearner, SLearner, XLearner])
    def test_predict_before_train_raises(self, ModelClass, dummy_data):
        X, _, _ = dummy_data
        model = ModelClass()
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict_uplift(X)

    @pytest.mark.parametrize("ModelClass", [TLearner, SLearner, XLearner])
    def test_shap_values_shape(self, ModelClass, dummy_data):
        X, t, y = dummy_data
        model = ModelClass()
        model.train(X, t, y)

        sample = X.iloc[:20]
        shap_vals = model.get_shap_values(sample)

        assert shap_vals.shape[0] == 20
        # S-Learner excludes the 'is_treated' column, so n_features == X.shape[1]
        assert shap_vals.shape[1] == X.shape[1]

    @pytest.mark.parametrize("ModelClass", [TLearner, SLearner, XLearner])
    def test_train_returns_self(self, ModelClass, dummy_data):
        X, t, y = dummy_data
        model = ModelClass()
        result = model.train(X, t, y)
        assert result is model


# ──────────────────────────────────────────────
# optimizer
# ──────────────────────────────────────────────

class TestOptimizeBudget:
    def test_selects_top_by_uplift(self, scored_df):
        res = optimize_budget(
            scored_df, "uplift", "outcome", "treatment",
            budget=1.5, cost_per_action=1.0, revenue_per_conversion=10.0,
        )
        assert res["selected_usrs"] == 1
        assert res["inc_conversions"] == pytest.approx(0.9)

    def test_respects_budget_limit(self, scored_df):
        res = optimize_budget(
            scored_df, "uplift", "outcome", "treatment",
            budget=2.5, cost_per_action=1.0, revenue_per_conversion=10.0,
        )
        assert res["selected_usrs"] == 2
        assert res["inc_conversions"] == pytest.approx(1.4)

    def test_full_budget_selects_all(self, scored_df):
        res = optimize_budget(
            scored_df, "uplift", "outcome", "treatment",
            budget=100.0, cost_per_action=1.0, revenue_per_conversion=10.0,
        )
        assert res["selected_usrs"] == len(scored_df)

    def test_profit_calculation(self, scored_df):
        res = optimize_budget(
            scored_df, "uplift", "outcome", "treatment",
            budget=2.0, cost_per_action=1.0, revenue_per_conversion=100.0,
        )
        # 2 users: uplift sum = 0.9 + 0.5 = 1.4
        # revenue = 1.4 * 100 = 140, cost = 2, profit = 138
        assert res["profit"] == pytest.approx(138.0)

    def test_zero_budget_selects_nobody(self, scored_df):
        res = optimize_budget(
            scored_df, "uplift", "outcome", "treatment",
            budget=0.0, cost_per_action=1.0, revenue_per_conversion=10.0,
        )
        assert res["selected_usrs"] == 0
        assert res["cost"] == 0.0

    def test_raises_on_invalid_cost(self, scored_df):
        with pytest.raises(ValueError, match="cost_per_action"):
            optimize_budget(
                scored_df, "uplift", "outcome", "treatment",
                budget=10.0, cost_per_action=0.0,
            )

    def test_raises_on_missing_columns(self, scored_df):
        with pytest.raises(ValueError, match="Missing columns"):
            optimize_budget(
                scored_df, "nonexistent", "outcome", "treatment",
                budget=10.0,
            )

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
