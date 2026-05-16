"""
Home page — hero section, live stats, project overview.
This file is the app entry point (shown as 'Home' in the sidebar).
"""

import streamlit as st
from utils.data_loader import load_data, train_models
from utils.styles import inject_css, card, section_header
from utils.viz_utils import plot_tier_donut, plot_value_distribution

st.set_page_config(
    page_title="FIFA Scout Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Sidebar branding ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
<div style='text-align:center;padding:0.8rem 0 1.4rem;'>
    <div style='font-size:2.2rem;'>⚽</div>
    <div style='font-size:1.1rem;font-weight:700;color:#f1f5f9;letter-spacing:0.03em;'>
        FIFA Scout AI
    </div>
    <div style='font-size:0.75rem;color:#94a3b8;margin-top:4px;'>
        Player Intelligence Platform
    </div>
</div>
<hr style='border-color:rgba(255,255,255,0.06);margin:0 0 1rem;'/>
""",
        unsafe_allow_html=True,
    )

# ── Load data + model ─────────────────────────────────────────────────────────
with st.spinner("Loading data and training models — this takes ~20 seconds on first run …"):
    df, cols = load_data()
    model_data = train_models()
m = model_data["metrics"]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style='padding:2.2rem 0 1rem;'>
    <div style='font-size:0.78rem;font-weight:600;color:#10b981;
                letter-spacing:0.1em;text-transform:uppercase;
                margin-bottom:0.6rem;'>
        AI-POWERED FOOTBALL ANALYTICS
    </div>
    <h1 style='font-size:2.4rem;font-weight:700;line-height:1.2;margin:0 0 0.7rem;'>
        FIFA Scout Intelligence<br>
        <span style='color:#10b981;'>Platform</span>
    </h1>
    <p style='color:#94a3b8;font-size:1.05rem;max-width:580px;line-height:1.6;margin:0;'>
        Machine-learning pipeline trained on 19,667 players.
        Predict market value, assign performance tiers, and
        discover similar players — the way professional clubs do it.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Players", f"{m['n_players']:,}")
with c2:
    st.metric("Regression R²", f"{m['r2']:.3f}")
with c3:
    st.metric("Value MAE", f"${m['mae']:.2f}M")
with c4:
    st.metric("Tier Accuracy", f"{m['accuracy']*100:.1f}%")
with c5:
    st.metric("Features", str(m["n_features"]))

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column overview ───────────────────────────────────────────────────────
left, right = st.columns([1.35, 1], gap="large")

with left:
    section_header("Dataset overview", "FIFA player statistics — cleaned and engineered")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Training set", f"{m['n_train']:,} players")
        st.metric("CV R² (mean ± std)", f"{m['cv_r2_mean']:.3f} ± {m['cv_r2_std']:.3f}")
    with col_b:
        st.metric("Test set", f"{m['n_test']:,} players")
        st.metric("CV Accuracy", f"{m['cv_acc_mean']*100:.1f}% ± {m['cv_acc_std']*100:.1f}%")

    st.plotly_chart(
        plot_value_distribution(df, cols["value"]),
        use_container_width=True,
    )

with right:
    section_header("Performance tiers", "Percentile-based tier segmentation")
    st.plotly_chart(plot_tier_donut(df), use_container_width=True)

    tier_counts = df["tier"].value_counts()
    for tier in ["Elite", "Good", "Average", "Developing"]:
        n = tier_counts.get(tier, 0)
        pct = n / len(df) * 100
        from utils.config import TIER_COLORS
        color = TIER_COLORS[tier]
        st.markdown(
            f"""
<div style='display:flex;justify-content:space-between;align-items:center;
            padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);'>
    <span style='color:{color};font-weight:600;font-size:0.85rem;'>{tier}</span>
    <span style='color:#94a3b8;font-size:0.82rem;'>{n:,} players ({pct:.1f}%)</span>
</div>""",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Feature cards ─────────────────────────────────────────────────────────────
section_header("Platform features", "What you can do with this tool")
st.markdown("<br>", unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)
features_data = [
    ("💰", "Player Valuation", "Enter any player profile and receive a precise market value prediction in seconds."),
    ("📊", "Analytics Dashboard", "Explore interactive charts across positions, ages, tiers, and countries."),
    ("🔍", "Similar Players", "Find the 5 most statistically similar players using cosine similarity."),
    ("🧠", "Model Insights", "Understand the preprocessing pipeline, feature importance, and model comparisons."),
]
for col, (icon, title, desc) in zip([f1, f2, f3, f4], features_data):
    with col:
        card(f"""
<div style='text-align:center;'>
    <div style='font-size:1.8rem;margin-bottom:10px;'>{icon}</div>
    <div style='font-weight:600;font-size:0.95rem;margin-bottom:6px;'>{title}</div>
    <div style='color:#94a3b8;font-size:0.8rem;line-height:1.5;'>{desc}</div>
</div>
""", padding="1.4rem 1rem")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("⚽ FIFA Scout AI — Built for a machine learning university project · Spring 2026")
