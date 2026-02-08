"""
Budget optimisation logic for Uplift Campaigns.

Provides a greedy targeting strategy (rank-by-uplift) and a
profitability-curve visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────


@dataclass
class OptimizationResult:
    """Structured output of :func:`optimize_budget`.

    Attributes:
        selected_users: Number of users in the target cohort.
        total_users: Total population size.
        cost: Total campaign cost (``selected_users * cost_per_action``).
        revenue: Projected revenue from incremental conversions.
        profit: ``revenue - cost``.
        roi_pct: Return on investment as a percentage.
        incremental_conversions: Sum of predicted uplift in cohort.
        avg_uplift: Mean uplift score in cohort.
        min_uplift_threshold: Lowest uplift score included.
        cohort_df: DataFrame of selected users.
    """

    selected_users: int
    total_users: int
    cost: float
    revenue: float
    profit: float
    roi_pct: float
    incremental_conversions: float
    avg_uplift: float
    min_uplift_threshold: float
    cohort_df: pd.DataFrame = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation (for legacy callers)."""
        return {
            "selected_usrs": self.selected_users,
            "total_usrs": self.total_users,
            "cost": self.cost,
            "revenue": self.revenue,
            "profit": self.profit,
            "roi": self.roi_pct,
            "inc_conversions": self.incremental_conversions,
            "avg_uplift": self.avg_uplift,
            "min_uplift_threshold": self.min_uplift_threshold,
            "cohort_df": self.cohort_df,
        }


# ──────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────


def optimize_budget(
    df: pd.DataFrame,
    uplift_col: str,
    outcome_col: str,
    treatment_col: str,
    budget: float,
    cost_per_action: float = 2.0,
    revenue_per_conversion: float = 50.0,
) -> dict[str, Any]:
    """Select customers greedily by descending uplift until the budget is spent.

    Args:
        df: Scored DataFrame (must contain *uplift_col*).
        uplift_col: Column holding predicted uplift scores.
        outcome_col: Column holding observed outcomes (unused for selection
            but kept in the result for downstream analysis).
        treatment_col: Column holding the treatment indicator.
        budget: Maximum spend.
        cost_per_action: Cost of targeting a single user.
        revenue_per_conversion: Revenue earned per incremental conversion.

    Returns:
        Dictionary of campaign metrics and the selected cohort DataFrame.

    Raises:
        ValueError: If *cost_per_action* is non-positive or required columns
            are missing.
    """
    if cost_per_action <= 0:
        raise ValueError("cost_per_action must be positive.")

    missing = {uplift_col, outcome_col, treatment_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    ranked = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)

    max_affordable = int(budget // cost_per_action)
    cutoff = min(max_affordable, len(ranked))
    cohort = ranked.iloc[:cutoff]

    inc_conv: float = float(cohort[uplift_col].sum())
    cost = cutoff * cost_per_action
    revenue = inc_conv * revenue_per_conversion
    profit = revenue - cost

    result = OptimizationResult(
        selected_users=cutoff,
        total_users=len(ranked),
        cost=cost,
        revenue=revenue,
        profit=profit,
        roi_pct=(profit / cost) * 100 if cost > 0 else 0.0,
        incremental_conversions=inc_conv,
        avg_uplift=float(cohort[uplift_col].mean()) if cutoff > 0 else 0.0,
        min_uplift_threshold=float(cohort[uplift_col].min()) if cutoff > 0 else 0.0,
        cohort_df=cohort,
    )
    return result.as_dict()


# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────


def plot_roi_curve(
    df: pd.DataFrame,
    uplift_col: str,
    cost_per_action: float,
    revenue_per_conversion: float,
) -> go.Figure:
    """Plot the projected profit as a function of population targeted.

    Args:
        df: Scored DataFrame.
        uplift_col: Column with predicted uplift scores.
        cost_per_action: Cost per targeted user.
        revenue_per_conversion: Revenue per incremental conversion.

    Returns:
        Plotly ``Figure`` object.
    """
    sorted_uplift = df[uplift_col].sort_values(ascending=False).reset_index(drop=True)
    n = len(sorted_uplift)

    users = np.arange(1, n + 1)
    cum_uplift = sorted_uplift.cumsum().values
    profits = cum_uplift * revenue_per_conversion - users * cost_per_action

    # Optimal targeting point
    max_idx = int(np.argmax(profits))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=users,
            y=profits,
            mode="lines",
            name="Projected Profit",
            line=dict(color="#22c55e", width=3),
        )
    )

    fig.add_annotation(
        x=users[max_idx],
        y=profits[max_idx],
        text=(
            f"Optimal: {users[max_idx]:,} users"
            f"<br>Profit: ${profits[max_idx]:,.0f}"
        ),
        showarrow=True,
        arrowhead=1,
    )

    fig.update_layout(
        title="Profitability Curve (Greedy Targeting)",
        xaxis_title="Number of Users Targeted",
        yaxis_title="Projected Profit ($)",
        template="plotly_white",
    )
    return fig
