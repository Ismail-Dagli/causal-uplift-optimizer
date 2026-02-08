"""
Streamlit Dashboard — Causal Uplift Optimizer Pro.

Targets *Persuadables* using T/S/X meta-learners, visualises ROI,
and computes SHAP-based explanations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from src.data_loader import generic_preprocess, load_hillstrom, preprocess
from src.model import BaseUpliftModel, SLearner, TLearner, XLearner
from src.optimizer import optimize_budget, plot_roi_curve

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Causal Uplift Optimizer Pro",
    page_icon="🧠",
    layout="wide",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type[BaseUpliftModel]] = {
    "T-Learner": TLearner,
    "S-Learner": SLearner,
    "X-Learner": XLearner,
}

_MAX_SHAP_SAMPLES = 500
_HIGH_CARDINALITY_THRESHOLD = 20
_TOP_CATEGORIES = 20


# ──────────────────────────────────────────────
# Cached helpers
# ──────────────────────────────────────────────


@st.cache_data(show_spinner="Loading example data…")
def _load_example_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Download / generate the Hillstrom dataset and preprocess it."""
    raw = load_hillstrom()
    X, treat, out = preprocess(raw)
    return raw, X, treat, out


@st.cache_resource(show_spinner="Training model…")
def _train_model(
    model_name: str,
    _X: pd.DataFrame,
    _t: pd.Series,
    _y: pd.Series,
) -> BaseUpliftModel:
    """Instantiate and train the selected uplift model."""
    cls = _MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name!r}")
    return cls().train(_X, _t, _y)


# ──────────────────────────────────────────────
# Sidebar — data source
# ──────────────────────────────────────────────

st.sidebar.title("🧠 Uplift Pro")

data_source = st.sidebar.radio("Data Source", ["Example Data", "Upload CSV"])

raw_df: pd.DataFrame | None = None
X: pd.DataFrame | None = None
treatment: pd.Series | None = None
outcome: pd.Series | None = None

if data_source == "Example Data":
    raw_df, X, treatment, outcome = _load_example_data()

else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is None:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

    raw_df = pd.read_csv(uploaded_file)

    if raw_df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    st.info(f"Loaded **{len(raw_df):,}** rows, **{len(raw_df.columns)}** columns.")

    all_cols = list(raw_df.columns)

    with st.sidebar.expander("Column Mapping", expanded=True):
        st.caption("Map your columns to the model requirements:")

        treat_col = st.selectbox(
            "Treatment Column (binary 0/1)",
            all_cols,
            help="Must be binary. E.g., 'Received_Email', 'Coupon_Sent'.",
        )
        outcome_col = st.selectbox(
            "Outcome Column (conversion)",
            [c for c in all_cols if c != treat_col],
            help="Binary conversion or continuous revenue.",
        )

        remaining = [c for c in all_cols if c not in {treat_col, outcome_col}]
        feat_cols = st.multiselect(
            "Feature Columns",
            remaining,
            default=remaining[: min(5, len(remaining))],
            help="User attributes used to predict persuadability.",
        )

    if st.sidebar.button("Process Data"):
        if raw_df[treat_col].nunique() > 2:
            st.error(
                f"Treatment column **{treat_col}** is not binary — "
                f"it has {raw_df[treat_col].nunique()} unique values."
            )
        else:
            try:
                X, treatment, outcome = generic_preprocess(
                    raw_df, treat_col, outcome_col, feat_cols
                )
                st.session_state["custom_data"] = (X, treatment, outcome)
                st.success("Data processed successfully!")
            except Exception as exc:
                st.error(f"Error processing data: {exc}")

    if "custom_data" in st.session_state:
        X, treatment, outcome = st.session_state["custom_data"]


# ──────────────────────────────────────────────
# Guard — stop early if no data is ready
# ──────────────────────────────────────────────

if X is None:
    st.stop()

# ──────────────────────────────────────────────
# Sidebar — model selection & training
# ──────────────────────────────────────────────

selected_model_name: str = st.sidebar.selectbox(
    "Choose Uplift Model",
    list(_MODEL_REGISTRY),
    index=0,
)

model = _train_model(selected_model_name, X, treatment, outcome)
uplift_scores: np.ndarray = model.predict_uplift(X)

df_scored: pd.DataFrame = raw_df.loc[X.index].copy()
df_scored["uplift_score"] = uplift_scores
df_scored["treatment"] = treatment.values
df_scored["outcome"] = outcome.values

# ──────────────────────────────────────────────
# Main tabs
# ──────────────────────────────────────────────

tab_perf, tab_opt, tab_explain = st.tabs(
    ["📊 Model Performance", "💰 Budget Optimizer", "🕵️ Explainability"]
)

# ──────────────────────────────────────────────
# Tab 1 — Performance
# ──────────────────────────────────────────────

with tab_perf:
    st.header(f"Performance: {selected_model_name}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Uplift (ATE)", f"{uplift_scores.mean():.2%}")
    col2.metric("Max Uplift", f"{uplift_scores.max():.2%}")
    col3.metric("Min Uplift", f"{uplift_scores.min():.2%}")

    # Decile analysis
    st.subheader("Uplift by Decile")
    num_deciles = min(10, len(uplift_scores) // 10)
    if num_deciles < 2:
        st.warning("Not enough data for decile analysis.")
    else:
        decile_labels = (
            pd.qcut(uplift_scores, q=num_deciles, labels=False, duplicates="drop") + 1
        )
        decile_df = (
            pd.DataFrame({"decile": decile_labels, "uplift": uplift_scores})
            .groupby("decile")["uplift"]
            .mean()
            .reset_index()
        )
        fig_d = px.bar(
            decile_df,
            x="decile",
            y="uplift",
            color="uplift",
            color_continuous_scale="RdYlGn",
            title="Average Uplift per Decile",
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # Attribute drill-down
    st.subheader("Attribute Drill-down")
    ignored_cols = {"uplift_score", "treatment", "outcome"}
    feature_options = [c for c in df_scored.columns if c not in ignored_cols]

    feature_to_plot = st.selectbox("Compare Uplift by Feature", feature_options)

    if feature_to_plot:
        plot_df = df_scored.copy()
        if plot_df[feature_to_plot].nunique() > _HIGH_CARDINALITY_THRESHOLD:
            if pd.api.types.is_numeric_dtype(plot_df[feature_to_plot]):
                try:
                    plot_df[feature_to_plot] = pd.qcut(
                        plot_df[feature_to_plot], q=10, duplicates="drop"
                    ).astype(str)
                except ValueError:
                    pass  # fallback to raw values if binning fails
            else:
                top = plot_df[feature_to_plot].value_counts().head(_TOP_CATEGORIES).index
                plot_df = plot_df[plot_df[feature_to_plot].isin(top)]

        agg = (
            plot_df.groupby(feature_to_plot)["uplift_score"]
            .mean()
            .reset_index()
            .sort_values("uplift_score")
        )
        fig_feat = px.bar(
            agg,
            x=feature_to_plot,
            y="uplift_score",
            title=f"Uplift by {feature_to_plot}",
            color="uplift_score",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_feat, use_container_width=True)


# ──────────────────────────────────────────────
# Tab 2 — Budget Optimisation
# ──────────────────────────────────────────────

with tab_opt:
    st.header("Campaign Budget Optimization")
    st.markdown("Adjust the sliders below to simulate different spend scenarios.")

    col_l, col_r = st.columns(2)
    with col_l:
        budget_input = st.number_input("Total Budget ($)", value=20_000, step=1_000)
        cost_per_email = st.number_input("Cost per Action ($)", value=1.5, step=0.1)
    with col_r:
        rev_per_con = st.number_input(
            "Revenue per Conversion ($)", value=75.0, step=5.0
        )

    if st.button("Optimize Campaign"):
        res = optimize_budget(
            df_scored,
            "uplift_score",
            "outcome",
            "treatment",
            budget=budget_input,
            cost_per_action=cost_per_email,
            revenue_per_conversion=rev_per_con,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Users to Target", f"{res['selected_usrs']:,}")
        m2.metric("Est. Incremental Conv.", f"{int(res['inc_conversions'])}")
        m3.metric("Projected Profit", f"${res['profit']:,.0f}")
        m4.metric("ROI", f"{res['roi']:.1f}%")

        fig_roi = plot_roi_curve(
            df_scored, "uplift_score", cost_per_email, rev_per_con
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        with st.expander("Show Target List"):
            st.dataframe(res["cohort_df"].head(100))


# ──────────────────────────────────────────────
# Tab 3 — SHAP Explainability
# ──────────────────────────────────────────────

with tab_explain:
    st.header("SHAP Explainability")
    st.info(
        f"SHAP values are computed on a random sample of up to "
        f"**{_MAX_SHAP_SAMPLES}** rows for performance."
    )

    if st.button("Calculate SHAP Values"):
        with st.spinner("Computing SHAP…"):
            X_sample = X.sample(min(_MAX_SHAP_SAMPLES, len(X)), random_state=42)

            try:
                shap_values = model.get_shap_values(X_sample)

                st.subheader("Global Feature Importance (SHAP)")
                fig_imp, _ = plt.subplots()
                shap.summary_plot(
                    shap_values, X_sample, plot_type="bar", show=False
                )
                plt.xlabel("Average |SHAP value|")
                st.pyplot(fig_imp)

                st.subheader("Feature Impact Direction")
                fig_dir, _ = plt.subplots()
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig_dir)

            except Exception as exc:
                st.error(f"Error computing SHAP values: {exc}")
