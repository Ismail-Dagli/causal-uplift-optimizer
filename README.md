# Causal Uplift Optimizer 🎯

A production-ready Causal Inference dashboard to identify "Persuadable" customers and optimize marketing budgets.

This application uses machine learning (Uplift Modeling) to estimate the Conditional Average Treatment Effect (CATE) of marketing campaigns. It helps you target the right users to maximize ROI and avoid wasting budget on "Sleeping Dogs" (users who react negatively to marketing) or "Sure Things" (users who buy anyway).

## 🌟 Features

- **Advanced Causal Models**:
  - **T-Learner**: Traditional two-model approach (Treatment vs Control).
  - **S-Learner**: Single model with treatment interaction.
  - **X-Learner**: State-of-the-art meta-learner for unbalanced data.
- **Budget Optimization Engine**: Interactive slider to find the optimal campaign size and projected ROI based on cost/revenue constraints.
- **Explainability**: SHAP (SHapley Additive exPlanations) values to understand *why* a customer is persuadable.
- **Interactive Visualization**: Plotly charts for Uplift by Decile, Cumulative Gain Curves (AUUC), and feature slicing.

## 🛠️ Tech Stack

- **Python 3.13+**
- **Streamlit**: Interactive Dashboard
- **XGBoost**: Gradient Boosting for CATE estimation
- **SHAP**: Model Interpretability
- **Plotly**: Data Visualization

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Ismail-Dagli/causal-uplift-optimizer.git
cd causal-uplift-optimizer
```

### 2. Install & Run with `uv` (Recommended)
This project uses [uv](https://github.com/astral-sh/uv) for fast package management.

```bash
# Install dependencies and run immediately
uv run streamlit run app.py
```

### Alternative: Using pip
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Alternative: Docker
```bash
docker build -t causal-uplift .
docker run -p 8501:8501 causal-uplift
```

## 📊 Dataset
The project automatically loads the **Kevin Hillstrom Email Marketing Dataset**.
- If internet is available, it downloads directly from the source.
- If offline, it generates a synthetic version with encoded causal effects.

## 🧠 Methodology within the App

1.  **Select a Model** (T, S, or X-Learner) to train on the data.
2.  **Analyze Performance**: Check the Uplift by Decile chart. A good model shows a steep downward slope.
3.  **Optimize Budget**: Go to the "Budget Optimizer" tab. Adjust your budget to see how many "Persuadables" you can afford and the resulting Profit.
4.  **Explain**: Use the "Explainability" tab to see which features (e.g., `history`, `recency`) drive positive uplift.

## 📄 License
MIT
