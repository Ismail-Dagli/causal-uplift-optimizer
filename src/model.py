"""
Causal Uplift Models: T-Learner, S-Learner, and X-Learner.

Each learner wraps XGBoost estimators and exposes a unified interface
for training, predicting uplift (CATE), and computing SHAP explanations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier, XGBRegressor

# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────

DEFAULT_LEARNER_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "n_estimators": 100,
    "learning_rate": 0.1,
    "random_state": 42,
    "eval_metric": "logloss",
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _extract_class1_shap(shap_values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Return SHAP values for the positive class.

    Older SHAP versions return a list ``[class0, class1]`` for binary
    classifiers, while newer versions return a single array.

    Args:
        shap_values: Raw output from ``TreeExplainer.shap_values``.

    Returns:
        2-D array of shape ``(n_samples, n_features)``.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    return np.asarray(shap_values)


def _ensure_trained(model: "BaseUpliftModel") -> None:
    """Raise if the model has not been trained yet."""
    if not model._is_trained:
        raise RuntimeError(
            f"{type(model).__name__} has not been trained. "
            "Call .train() before .predict_uplift() or .get_shap_values()."
        )


# ──────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────


class BaseUpliftModel(ABC):
    """Abstract base class for all meta-learner uplift models.

    Args:
        learner_params: XGBoost hyper-parameters passed to every
            internal estimator.  Defaults to :data:`DEFAULT_LEARNER_PARAMS`.
    """

    def __init__(self, learner_params: dict[str, Any] | None = None) -> None:
        self.learner_params: dict[str, Any] = (
            learner_params or DEFAULT_LEARNER_PARAMS.copy()
        )
        self._is_trained: bool = False

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> BaseUpliftModel:
        """Fit the model on labelled data."""

    @abstractmethod
    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row uplift (CATE) estimates."""

    @abstractmethod
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return SHAP values explaining the uplift predictions."""


# ──────────────────────────────────────────────
# T-Learner
# ──────────────────────────────────────────────


class TLearner(BaseUpliftModel):
    """Two-model learner: one classifier per treatment arm.

    Uplift = P(Y=1 | X, T=1)  -  P(Y=1 | X, T=0).
    SHAP is derived from the treatment-arm model as a proxy.
    """

    def __init__(self, learner_params: dict[str, Any] | None = None) -> None:
        super().__init__(learner_params)
        self.model_treatment = XGBClassifier(**self.learner_params)
        self.model_control = XGBClassifier(**self.learner_params)

    def train(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> TLearner:
        treat_mask = treatment == 1
        self.model_treatment.fit(X.loc[treat_mask], outcome[treat_mask])
        self.model_control.fit(X.loc[~treat_mask], outcome[~treat_mask])
        self._is_trained = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        _ensure_trained(self)
        p_treat = self.model_treatment.predict_proba(X)[:, 1]
        p_ctrl = self.model_control.predict_proba(X)[:, 1]
        return p_treat - p_ctrl

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values from the treatment-group model (positive-class)."""
        _ensure_trained(self)
        explainer = shap.TreeExplainer(self.model_treatment)
        return _extract_class1_shap(explainer.shap_values(X))


# ──────────────────────────────────────────────
# S-Learner
# ──────────────────────────────────────────────


class SLearner(BaseUpliftModel):
    """Single-model learner: treatment indicator is an input feature.

    Uplift is obtained by predicting twice — once with T=1, once with T=0 —
    and differencing.
    """

    def __init__(self, learner_params: dict[str, Any] | None = None) -> None:
        super().__init__(learner_params)
        self.model = XGBClassifier(**self.learner_params)

    def train(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> SLearner:
        X_aug = X.assign(is_treated=treatment.values)
        self.model.fit(X_aug, outcome)
        self._is_trained = True
        return self

    def _predict_proba_with_flag(
        self, X: pd.DataFrame, treated: int
    ) -> np.ndarray:
        """Predict P(Y=1) with a fixed treatment flag."""
        return self.model.predict_proba(X.assign(is_treated=treated))[:, 1]

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        _ensure_trained(self)
        return (
            self._predict_proba_with_flag(X, treated=1)
            - self._predict_proba_with_flag(X, treated=0)
        )

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP uplift = SHAP(T=1) - SHAP(T=0), excluding the treatment flag."""
        _ensure_trained(self)
        explainer = shap.TreeExplainer(self.model)

        shap_treat = _extract_class1_shap(
            explainer.shap_values(X.assign(is_treated=1))
        )
        shap_ctrl = _extract_class1_shap(
            explainer.shap_values(X.assign(is_treated=0))
        )

        # Drop the last column (``is_treated``) so dimensions match X
        return (shap_treat - shap_ctrl)[:, :-1]


# ──────────────────────────────────────────────
# X-Learner
# ──────────────────────────────────────────────


class XLearner(BaseUpliftModel):
    """X-Learner: two-stage CATE estimator.

    Stage 1 — Outcome models per arm (classifiers).
    Stage 2 — Imputed treatment-effect models (regressors).
    Final CATE is a propensity-weighted average of the two effect
    estimators (defaults to equal weight for RCT data).
    """

    def __init__(self, learner_params: dict[str, Any] | None = None) -> None:
        super().__init__(learner_params)

        # Stage 1: outcome classifiers
        self.model_t = XGBClassifier(**self.learner_params)
        self.model_c = XGBClassifier(**self.learner_params)

        # Stage 2: effect regressors
        reg_params = {
            k: v
            for k, v in self.learner_params.items()
            if k != "eval_metric"
        }
        reg_params["objective"] = "reg:squarederror"
        self.tau_t = XGBRegressor(**reg_params)
        self.tau_c = XGBRegressor(**reg_params)

        self.propensity: float = 0.5  # RCT assumption

    def train(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> XLearner:
        treat_mask = treatment == 1

        # Stage 1
        self.model_t.fit(X.loc[treat_mask], outcome[treat_mask])
        self.model_c.fit(X.loc[~treat_mask], outcome[~treat_mask])

        # Stage 2 — imputed treatment effects
        d_treat = (
            outcome[treat_mask]
            - self.model_c.predict_proba(X.loc[treat_mask])[:, 1]
        )
        d_ctrl = (
            self.model_t.predict_proba(X.loc[~treat_mask])[:, 1]
            - outcome[~treat_mask]
        )

        self.tau_t.fit(X.loc[treat_mask], d_treat)
        self.tau_c.fit(X.loc[~treat_mask], d_ctrl)

        self._is_trained = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        _ensure_trained(self)
        g = self.propensity
        return g * self.tau_c.predict(X) + (1 - g) * self.tau_t.predict(X)

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted-average SHAP from both CATE estimators."""
        _ensure_trained(self)
        shap_t = shap.TreeExplainer(self.tau_t).shap_values(X)
        shap_c = shap.TreeExplainer(self.tau_c).shap_values(X)

        g = self.propensity
        return np.asarray(shap_c) * g + np.asarray(shap_t) * (1 - g)
