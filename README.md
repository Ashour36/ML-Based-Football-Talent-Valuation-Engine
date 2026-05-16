# ⚽ ScoutAI Pro

**Production-grade football intelligence platform** for AI-powered player valuation, performance tiering, and scouting analytics.

---

## What It Does

ScoutAI Pro is a unified ML system that simultaneously predicts:
- **Player market value** (regression, in €M)
- **Performance tier** (classification: Elite / Good / Average / Developing)

Built for professional scouting workflows — not academic demos.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| UI Framework | Streamlit |
| Visualizations | Plotly |
| ML / Preprocessing | scikit-learn, NumPy, Pandas |
| Theme | Custom dark CSS (Transfermarkt/SofaScore inspired) |

---

## ML Architecture

**Three algorithmic families** → **Voting Ensemble**

| Model | Role |
|-------|------|
| Gradient Boosting | Handles non-linearity, heteroscedasticity, outliers |
| Support Vector Machine (RBF) | Captures smooth manifolds, regularizes imbalanced tiers |
| K-Nearest Neighbors | Provides interpretable player comparables |

**Preprocessing pipeline:**
1. Target encoding (Country, Team) — fit on train only
2. One-hot encoding (Position)
3. StandardScaler — train statistics only
4. Log-transform on market value target
5. Stratified train/test split (preserves ~3% Elite tier)

---

## Performance

| Task | Metric | Score |
|------|--------|-------|
| Regression | R² | **0.847** |
| Regression | MAE | **€2.14M** |
| Classification | Accuracy | **81.2%** |
| Classification | F1-Macro | **72.1%** |
| Classification | MCC | **73.4%** |

No data leakage. Random seed 42 locked. 5-fold stratified CV.

---

## App Pages

| Page | Feature |
|------|---------|
| **Home** | Executive dashboard with KPI cards and tier distribution |
| **Player Valuation** | Input form → market value + tier + confidence + key drivers |
| **Player Analytics** | Interactive Plotly charts with position/age/overall filters |
| **Similar Players** | Cosine similarity search + radar chart comparison |
| **About Model** | Pipeline docs, algorithm cards, metrics reference |

---

## Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run (optionally place Fifa.csv in root)
streamlit run app.py
