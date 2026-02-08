"""
Data loader for the Kevin Hillstrom Email Marketing dataset.

Supports two modes:
  1. Built-in Hillstrom data (download or synthetic fallback).
  2. Generic CSV preprocessing for custom uploads.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HILLSTROM_URL = (
    "https://raw.githubusercontent.com/demandlib/causal-data/main/hillstrom.csv"
)

# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

_BRACKET_RE = r"[\[\]'\"]"
_XGBOOST_UNSAFE_RE = r"[\[\]<>]"


def _coerce_numeric_columns(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Attempt to convert object columns that contain numeric data to float.

    Handles dirty CSV artefacts such as ``"[1.02]"`` or ``"'3.5'"`` by
    stripping bracket/quote characters before conversion.

    Args:
        df: Input DataFrame (modified copy is returned).
        threshold: Minimum fraction of successfully-parsed values required
            to accept the conversion (0.0–1.0).

    Returns:
        DataFrame with cleaned numeric columns.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        # Fast path — try direct conversion without regex overhead
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() <= threshold:
            # Slow path — strip bracket/quote artefacts, then retry
            cleaned = df[col].astype(str).str.replace(_BRACKET_RE, "", regex=True)
            numeric = pd.to_numeric(cleaned, errors="coerce")

        if numeric.notna().mean() > threshold:
            df[col] = numeric.fillna(0.0)
    return df


def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove characters forbidden by XGBoost from column names.

    XGBoost rejects feature names containing ``[``, ``]``, or ``<``.

    Args:
        df: DataFrame whose columns will be sanitised in-place.

    Returns:
        Same DataFrame with cleaned column names.
    """
    df.columns = df.columns.astype(str).str.replace(_XGBOOST_UNSAFE_RE, "", regex=True)
    return df


def _one_hot_encode(
    df: pd.DataFrame,
    columns: list[str],
    drop_first: bool = False,
) -> pd.DataFrame:
    """One-hot encode specified columns (vectorised, no loop).

    Args:
        df: Input DataFrame.
        columns: Column names to encode.
        drop_first: Whether to drop the first dummy level.

    Returns:
        DataFrame with original columns replaced by dummy variables.
    """
    if not columns:
        return df
    return pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)


# ──────────────────────────────────────────────
# Synthetic data generation
# ──────────────────────────────────────────────


def _generate_synthetic_hillstrom(n: int = 42_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic version of the Hillstrom dataset.

    Args:
        n: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame mimicking the Hillstrom schema.
    """
    rng = np.random.default_rng(seed)

    segments = rng.choice(
        ["Womens E-Mail", "Mens E-Mail", "No E-Mail"],
        size=n,
        p=[0.33, 0.33, 0.34],
    )

    recency = rng.integers(1, 13, size=n)
    history = rng.exponential(scale=200, size=n).round(2)

    # Causal structure — treatment uplift
    is_treated = segments != "No E-Mail"
    base_prob = 0.05 + 0.02 * (history > 200) + 0.01 * (recency < 6)
    treat_boost = np.where(is_treated, 0.04, 0.0)
    treat_boost += np.where(is_treated & (history > 300) & (recency < 4), 0.06, 0.0)

    conversion_prob = np.clip(base_prob + treat_boost, 0, 1)
    conversion = rng.binomial(1, conversion_prob)

    df = pd.DataFrame(
        {
            "recency": recency,
            "history_segment": pd.cut(
                history,
                bins=[0, 100, 200, 350, 500, np.inf],
                labels=[
                    "1) $0 - $100",
                    "2) $100 - $200",
                    "3) $200 - $350",
                    "4) $350 - $500",
                    "5) $500 - $750",
                ],
            ).astype(str),
            "history": history,
            "mens": rng.integers(0, 2, size=n),
            "womens": rng.integers(0, 2, size=n),
            "zip_code": rng.choice(
                ["Urban", "Suburban", "Rural"], size=n, p=[0.47, 0.35, 0.18]
            ),
            "newbie": rng.integers(0, 2, size=n),
            "channel": rng.choice(
                ["Phone", "Web", "Multichannel"], size=n, p=[0.40, 0.35, 0.25]
            ),
            "segment": segments,
            "visit": (conversion | rng.binomial(1, 0.15, size=n)).clip(0, 1),
            "conversion": conversion,
            "spend": np.where(
                conversion == 1, rng.exponential(scale=30, size=n).round(2), 0.0
            ),
        }
    )
    return df


# ──────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────


def _try_download_hillstrom() -> pd.DataFrame | None:
    """Attempt to download the Hillstrom CSV from a public URL.

    Returns:
        DataFrame on success, ``None`` on any network / parse error.
    """
    try:
        with urllib.request.urlopen(HILLSTROM_URL, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        return pd.read_csv(io.StringIO(raw))
    except Exception:
        logger.info("Hillstrom download failed — falling back to synthetic data.")
        return None


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def load_hillstrom() -> pd.DataFrame:
    """Load the Hillstrom Email Marketing dataset.

    Tries a remote download first; falls back to synthetic generation.

    Returns:
        Raw Hillstrom DataFrame.
    """
    df = _try_download_hillstrom()
    return df if df is not None else _generate_synthetic_hillstrom()


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess the raw Hillstrom dataset into model-ready arrays.

    Args:
        df: Raw Hillstrom DataFrame (from :func:`load_hillstrom`).

    Returns:
        Tuple of ``(X, treatment, outcome)`` where *X* is one-hot encoded.
    """
    df = df.copy()

    treatment: pd.Series = (df["segment"] != "No E-Mail").astype(int)
    outcome: pd.Series = (df["conversion"] > 0).astype(int)

    feature_cols = ["recency", "history", "mens", "womens", "newbie"]
    categorical_cols = ["zip_code", "channel"]

    X = pd.concat([df[feature_cols], df[categorical_cols]], axis=1)
    X = _one_hot_encode(X, columns=categorical_cols, drop_first=False)

    return X, treatment, outcome


def generic_preprocess(
    df: pd.DataFrame,
    treat_col: str,
    outcome_col: str,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess a user-supplied CSV into model-ready arrays.

    Handles dirty data (stringified numbers, bracket artefacts) and
    automatically one-hot encodes remaining categorical columns.

    Args:
        df: Raw user DataFrame.
        treat_col: Name of the binary treatment column.
        outcome_col: Name of the outcome column.
        feature_cols: Explicit list of feature columns.  If ``None``,
            all columns except *treat_col* and *outcome_col* are used.

    Returns:
        Tuple of ``(X, treatment, outcome)``.

    Raises:
        ValueError: If the DataFrame is empty after dropping NaNs,
            or required columns are missing.
    """
    df = df.dropna().copy()

    if df.empty:
        raise ValueError("DataFrame is empty after dropping rows with missing values.")

    missing = {treat_col, outcome_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    treatment: pd.Series = df[treat_col].astype(int)
    outcome: pd.Series = df[outcome_col].astype(int)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {treat_col, outcome_col}]

    X: pd.DataFrame = df[feature_cols].copy()

    # 1. Clean dirty numeric columns (e.g. "[1.02]" → 1.02)
    X = _coerce_numeric_columns(X)

    # 2. One-hot encode remaining categoricals
    cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
    X = _one_hot_encode(X, columns=cat_cols, drop_first=True)

    # 3. Sanitise column names for XGBoost compatibility
    X = _sanitize_column_names(X)

    return X, treatment, outcome
