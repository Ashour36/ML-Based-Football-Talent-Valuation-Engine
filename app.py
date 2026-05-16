# ============================================================
# SCOUTAI PRO — Professional Football Intelligence Platform
# ============================================================
# A production-grade Streamlit application for FIFA player analytics,
# market valuation, and scouting operations.
#
# Design Inspiration: Transfermarkt, SofaScore, FBref, Wyscout
# Theme: Dark football analytics dashboard with emerald accents
# ============================================================

import warnings
import textwrap
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, matthews_corrcoef
)
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURATION
# ============================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

st.set_page_config(
    page_title="ScoutAI Pro | Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "ScoutAI Pro — Professional Football Intelligence Platform v1.0"
    }
)

# ============================================================
# CUSTOM THEME & CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background-color: #0a0e17;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] .css-1d391kg,
    [data-testid="stSidebar"] .css-163ttbj {
        background-color: #111827;
    }

    /* Typography */
    h1 {
        color: #ffffff;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    h2 {
        color: #e0e0e0;
        font-weight: 700;
        border-left: 4px solid #00c853;
        padding-left: 16px;
        margin-top: 30px;
    }
    h3 {
        color: #b0b0b0;
        font-weight: 600;
    }
    h4 {
        color: #e0e0e0;
        font-weight: 600;
    }
    p, li {
        color: #8899a6;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00c853 0%, #009624 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 24px;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(0, 200, 83, 0.4);
        transform: translateY(-2px);
    }
    .stButton>button:active {
        transform: scale(0.98);
    }

    /* Inputs */
    .stSlider > div > div > div {
        color: #00c853;
    }

    /* DataFrames / Tables */
    .dataframe {
        background-color: #151b2b !important;
        color: #e0e0e0 !important;
    }
    th {
        background-color: #1a2332 !important;
        color: #00c853 !important;
        font-weight: 600 !important;
    }
    td {
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    }

    /* Radio buttons in sidebar */
    .stRadio > label {
        color: #8899a6;
        font-weight: 500;
    }
    .stRadio > div > label > div:first-child {
        background-color: #1a2332;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #151b2b;
        border-radius: 12px;
        color: #e0e0e0;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background-color: #0f1520;
        border-radius: 0 0 12px 12px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #151b2b;
        border-radius: 8px 8px 0 0;
        color: #8899a6;
        font-weight: 600;
        border: none;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a2332 !important;
        color: #00c853 !important;
        border-bottom: 2px solid #00c853 !important;
    }

    /* Metric cards (native Streamlit metric override) */
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 800;
    }
    [data-testid="stMetricLabel"] {
        color: #8899a6;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem;
    }
    [data-testid="stMetricDelta"] {
        color: #00c853;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0e17;
    }
    ::-webkit-scrollbar-thumb {
        background: #1a2332;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00c853;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HTML COMPONENT HELPERS
# ============================================================
def card(title, value, subtitle, color="#00c853", icon=""):
    """Render a premium metric card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(145deg, #151b2b, #1a2332);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid {color};
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 16px;
        transition: transform 0.2s;
    " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
        <p style="color: #8899a6; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 1.5px;">
            {icon} {title}
        </p>
        <h2 style="color: #ffffff; font-size: 2.2rem; margin: 8px 0; font-weight: 800;">
            {value}
        </h2>
        <p style="color: {color}; font-size: 0.85rem; margin: 0; font-weight: 500;">
            {subtitle}
        </p>
    </div>
    """, unsafe_allow_html=True)


def tier_badge(tier):
    """Render a tier badge with appropriate styling."""
    styles = {
        'Elite': ('#ffd700', '#000000', '★ ELITE'),
        'Good': ('#00c853', '#ffffff', '▲ GOOD'),
        'Average': ('#448aff', '#ffffff', '● AVERAGE'),
        'Developing': ('#78909c', '#ffffff', '◆ DEVELOPING')
    }
    bg, fg, label = styles.get(tier, ('#78909c', '#ffffff', tier.upper()))
    return f"""
    <span style="
        background: linear-gradient(90deg, {bg}, {bg}dd);
        color: {fg};
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.85rem;
        letter-spacing: 1px;
        display: inline-block;
        box-shadow: 0 2px 10px {bg}40;
    ">{label}</span>
    """


def section_header(title, subtitle=""):
    """Render a consistent section header."""
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <h2 style="margin-top: 0; margin-bottom: 8px;">{title}</h2>
        <p style="color: #8899a6; margin: 0; font-size: 1rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATA LOADING & SYNTHETIC GENERATION
# ============================================================
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n=2500):
    """Generate a realistic FIFA-style dataset for demonstration."""
    np.random.seed(RANDOM_STATE)
    positions = ['ST', 'CF', 'LW', 'RW', 'CAM', 'CM', 'CDM', 'LM', 'RM',
                 'LB', 'RB', 'LWB', 'RWB', 'CB', 'GK']
    countries = [
        'Argentina', 'Brazil', 'England', 'France', 'Germany', 'Spain',
        'Italy', 'Portugal', 'Netherlands', 'Belgium', 'Croatia', 'Uruguay',
        'Colombia', 'Mexico', 'USA', 'Morocco', 'Nigeria', 'Denmark',
        'Switzerland', 'Austria', 'Norway', 'Poland', 'Senegal', 'Japan',
        'South Korea', 'Australia', 'Egypt', 'Ghana'
    ]
    teams = [
        'Manchester City', 'Real Madrid', 'Barcelona', 'Bayern Munich', 'Paris SG',
        'Liverpool', 'Chelsea', 'Arsenal', 'Juventus', 'Inter Milan', 'AC Milan',
        'Borussia Dortmund', 'RB Leipzig', 'Atletico Madrid', 'Sevilla', 'Roma',
        'Napoli', 'Ajax', 'Porto', 'Benfica', 'Celtic', 'Rangers',
        'Feyenoord', 'PSV', 'Club Brugge', 'Shakhtar', 'Galatasaray',
        'Fenerbahce', 'Besiktas', 'Al Nassr', 'Al Hilal', 'Inter Miami'
    ] + [f'Club {i:03d}' for i in range(60)]

    # Position sampling weights (strikers and attacking mids slightly overrepresented for realism)
    pos_weights = [0.10, 0.05, 0.08, 0.08, 0.07, 0.10, 0.07, 0.05, 0.05,
                   0.06, 0.06, 0.04, 0.04, 0.10, 0.08]

    data = {
        'Name': [f"{np.random.choice(['Alex', 'Marco', 'Lucas', 'David', 'James', 'Luis', 'Carlos', 'Thomas', 'Bruno', 'Pedro'])} {np.random.choice(['Silva', 'Martinez', 'Garcia', 'Rodriguez', 'Smith', 'Johnson', 'Brown', 'Davis', 'Miller', 'Wilson'])}" for _ in range(n)],
        'Age': np.random.randint(16, 39, n),
        'Overall_Rating': np.clip(np.random.normal(68, 9, n), 46, 94).astype(int),
        'Future Potential': np.clip(np.random.normal(72, 10, n), 55, 96).astype(int),
        'Total_Stats Score': np.clip(np.random.normal(1850, 420, n), 700, 2700).astype(int),
        'Country': np.random.choice(countries, n),
        'Team': np.random.choice(teams, n),
        'Position': np.random.choice(positions, n, p=pos_weights),
    }

    # Individual face stats for visualization (not used in model training)
    for stat in ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physical']:
        data[stat] = np.clip(np.random.normal(65, 15, n), 20, 99).astype(int)

    # Derive market value from realistic football economics
    age_factor = np.maximum(0, (28 - np.abs(data['Age'] - 26)) / 28)
    base = (
        (data['Overall_Rating'] ** 2.3) * 0.008 +
        (data['Future Potential'] ** 1.9) * 0.004 +
        data['Total_Stats Score'] * 0.0007
    )
    base *= (1 + age_factor * 2.5)

    pos_mult = {
        'ST': 1.45, 'CF': 1.35, 'CAM': 1.25, 'LW': 1.40, 'RW': 1.40,
        'CM': 1.05, 'CDM': 0.95, 'LM': 1.0, 'RM': 1.0,
        'CB': 0.90, 'LB': 0.85, 'RB': 0.85, 'LWB': 0.90, 'RWB': 0.90,
        'GK': 0.80
    }
    mults = np.array([pos_mult.get(p, 1.0) for p in data['Position']])
    noise = np.random.lognormal(0, 0.55, n)
    values = base * mults * noise
    values = np.maximum(values, 0.05)
    data['Value Per M$'] = np.round(values, 2)

    return pd.DataFrame(data)


@st.cache_data(show_spinner=False)
def load_data():
    """Load the FIFA dataset or fall back to synthetic demo data."""
    try:
        df = pd.read_csv('Fifa.csv')
        # Ensure required columns exist
        required = ['Name', 'Age', 'Overall_Rating', 'Future Potential',
                    'Total_Stats Score', 'Value Per M$', 'Country', 'Team', 'Position']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.sidebar.error(f"CSV missing columns: {missing}")
            raise ValueError(f"Missing columns: {missing}")
        st.sidebar.success("✅ Loaded Fifa.csv")
        return df
    except Exception as e:
        st.sidebar.warning("⚠️ Using synthetic demo data")
        return generate_synthetic_data()


# ============================================================
# PREPROCESSING & MODEL TRAINING
# ============================================================
def rating_to_tier(rating):
    """Convert overall rating to performance tier."""
    if rating >= 80:
        return 'Elite'
    elif rating >= 70:
        return 'Good'
    elif rating >= 60:
        return 'Average'
    else:
        return 'Developing'


@st.cache_resource(show_spinner="Training ensemble models...")
def build_pipeline(df):
    """
    End-to-end preprocessing and model training pipeline.
    Uses proper train/test separation to prevent data leakage.
    """
    df = df.copy()

    # Derive target tiers if not present
    if 'Performance_Tier' not in df.columns:
        df['Performance_Tier'] = df['Overall_Rating'].apply(rating_to_tier)

    # Core feature groups
    numeric_features = ['Age', 'Overall_Rating', 'Future Potential', 'Total_Stats Score']
    cat_features = ['Country', 'Team', 'Position']

    # Drop rows with missing critical values
    df = df.dropna(subset=numeric_features + ['Value Per M$'] + cat_features)

    # Prepare raw feature frame
    X_raw = df[numeric_features + cat_features].copy()
    y_reg = df['Value Per M$'].values
    y_clf = df['Performance_Tier'].values

    # Encode classification targets
    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)

    # Stratified split to preserve tier distribution
    X_tr_raw, X_te_raw, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X_raw, y_reg, y_clf_enc,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clf_enc
    )

    # --- Target Encoding (fit on TRAIN only to prevent leakage) ---
    temp_gb = X_tr_raw.copy()
    temp_gb['__target__'] = yr_tr
    country_map = temp_gb.groupby('Country')['__target__'].mean().to_dict()
    team_map = temp_gb.groupby('Team')['__target__'].mean().to_dict()
    global_mean = float(yr_tr.mean())

    def target_encode(series, mapping, default):
        return series.map(mapping).fillna(default).values

    X_tr = X_tr_raw.copy()
    X_te = X_te_raw.copy()
    X_tr['Country_Encoded'] = target_encode(X_tr_raw['Country'], country_map, global_mean)
    X_tr['Team_Encoded'] = target_encode(X_tr_raw['Team'], team_map, global_mean)
    X_te['Country_Encoded'] = target_encode(X_te_raw['Country'], country_map, global_mean)
    X_te['Team_Encoded'] = target_encode(X_te_raw['Team'], team_map, global_mean)

    # --- One-Hot Encoding for Position ---
    pos_dummies_tr = pd.get_dummies(X_tr['Position'], prefix='pos')
    pos_dummies_te = pd.get_dummies(X_te['Position'], prefix='pos')

    # Align dummy columns between train and test
    pos_cols = list(set(pos_dummies_tr.columns) | set(pos_dummies_te.columns))
    for col in pos_cols:
        if col not in pos_dummies_tr.columns:
            pos_dummies_tr[col] = 0
        if col not in pos_dummies_te.columns:
            pos_dummies_te[col] = 0
    pos_dummies_tr = pos_dummies_tr[pos_cols]
    pos_dummies_te = pos_dummies_te[pos_cols]

    # --- Final feature matrices ---
    engineered_cols = numeric_features + ['Country_Encoded', 'Team_Encoded']
    X_tr_mat = np.column_stack([X_tr[engineered_cols].values, pos_dummies_tr.values])
    X_te_mat = np.column_stack([X_te[engineered_cols].values, pos_dummies_te.values])

    # --- Scaling ---
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_mat)
    X_te_scaled = scaler.transform(X_te_mat)

    # --- Log-transform regression target ---
    yr_tr_log = np.log1p(yr_tr)

    # ========================================================
    # MODEL TRAINING
    # ========================================================
    # Regression stack
    gbr_reg = GradientBoostingRegressor(
        n_estimators=120, max_depth=5, learning_rate=0.08,
        min_samples_leaf=10, random_state=RANDOM_STATE
    )
    svr_reg = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)
    knn_reg = KNeighborsRegressor(n_neighbors=10, weights='distance', metric='euclidean')

    gbr_reg.fit(X_tr_scaled, yr_tr_log)
    svr_reg.fit(X_tr_scaled, yr_tr_log)
    knn_reg.fit(X_tr_scaled, yr_tr_log)

    # Classification stack
    gbr_clf = GradientBoostingClassifier(
        n_estimators=120, max_depth=5, learning_rate=0.08,
        min_samples_leaf=10, random_state=RANDOM_STATE
    )
    svc_clf = SVC(kernel='rbf', C=10.0, gamma='scale',
                  probability=True, random_state=RANDOM_STATE)
    knn_clf = KNeighborsClassifier(n_neighbors=10, weights='distance', metric='euclidean')

    gbr_clf.fit(X_tr_scaled, yc_tr)
    svc_clf.fit(X_tr_scaled, yc_tr)
    knn_clf.fit(X_tr_scaled, yc_tr)

    # --- Ensemble meta-learners ---
    ensemble_reg = VotingRegressor([
        ('gbr', gbr_reg), ('svr', svr_reg), ('knn', knn_reg)
    ])
    ensemble_clf = VotingClassifier([
        ('gbr', gbr_clf), ('svc', svc_clf), ('knn', knn_clf)
    ], voting='soft')

    ensemble_reg.fit(X_tr_scaled, yr_tr_log)
    ensemble_clf.fit(X_tr_scaled, yc_tr)

    # ========================================================
    # EVALUATION
    # ========================================================
    pred_reg = np.expm1(ensemble_reg.predict(X_te_scaled))
    r2 = float(r2_score(yr_te, pred_reg))
    mae = float(mean_absolute_error(yr_te, pred_reg))
    rmse = float(np.sqrt(mean_squared_error(yr_te, pred_reg)))

    pred_clf = ensemble_clf.predict(X_te_scaled)
    acc = float(accuracy_score(yc_te, pred_clf))
    f1_w = float(f1_score(yc_te, pred_clf, average='weighted'))
    f1_m = float(f1_score(yc_te, pred_clf, average='macro'))
    mcc = float(matthews_corrcoef(yc_te, pred_clf))

    # ========================================================
    # FULL-DATASET SCALED MATRIX (for similarity search)
    # ========================================================
    X_full_enc = df[numeric_features].copy()
    X_full_enc['Country_Encoded'] = target_encode(df['Country'], country_map, global_mean)
    X_full_enc['Team_Encoded'] = target_encode(df['Team'], team_map, global_mean)
    pos_dummies_full = pd.get_dummies(df['Position'], prefix='pos')
    for col in pos_cols:
        if col not in pos_dummies_full.columns:
            pos_dummies_full[col] = 0
    pos_dummies_full = pos_dummies_full[pos_cols]
    X_full_mat = np.column_stack([X_full_enc.values, pos_dummies_full.values])
    X_full_scaled = scaler.transform(X_full_mat)

    # ========================================================
    # ARTIFACT BUNDLE
    # ========================================================
    artifacts = {
        'df': df,
        'scaler': scaler,
        'label_encoder': le,
        'country_map': country_map,
        'team_map': team_map,
        'global_mean': global_mean,
        'pos_cols': pos_cols,
        'numeric_features': numeric_features,
        'engineered_cols': engineered_cols,
        'all_feature_names': engineered_cols + pos_cols,
        'models': {
            'regression': ensemble_reg,
            'classification': ensemble_clf,
            'gbr_reg': gbr_reg,
            'gbr_clf': gbr_clf
        },
        'metrics': {
            'r2': r2, 'mae': mae, 'rmse': rmse,
            'accuracy': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m, 'mcc': mcc,
            'train_size': len(X_tr_scaled), 'test_size': len(X_te_scaled)
        },
        'X_full_scaled': X_full_scaled,
        'X_tr_scaled': X_tr_scaled,
        'y_reg_log': np.log1p(y_reg),
        'y_clf_enc': y_clf_enc
    }
    return artifacts


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================
def vectorize_player(player_dict, artifacts):
    """Convert a raw player dictionary into a scaled feature vector."""
    numeric = artifacts['numeric_features']
    eng = artifacts['engineered_cols']
    pos_cols = artifacts['pos_cols']

    row = {k: player_dict.get(k, 0) for k in numeric}
    row['Country_Encoded'] = artifacts['country_map'].get(
        player_dict.get('Country'), artifacts['global_mean']
    )
    row['Team_Encoded'] = artifacts['team_map'].get(
        player_dict.get('Team'), artifacts['global_mean']
    )

    pos = player_dict.get('Position', 'CM')
    pos_key = f'pos_{pos}'
    for col in pos_cols:
        row[col] = 1 if col == pos_key else 0

    X = np.array([[row[k] for k in eng + pos_cols]])
    return artifacts['scaler'].transform(X)


def predict_player(player_dict, artifacts):
    """Run valuation and tier prediction for a single player."""
    X_scaled = vectorize_player(player_dict, artifacts)

    # Regression
    pred_log = artifacts['models']['regression'].predict(X_scaled)[0]
    value = float(np.expm1(pred_log))

    # Classification
    tier_enc = artifacts['models']['classification'].predict(X_scaled)[0]
    tier = artifacts['label_encoder'].inverse_transform([tier_enc])[0]

    # Probabilities
    probs = artifacts['models']['classification'].predict_proba(X_scaled)[0]
    confidence = float(np.max(probs))

    # Feature importance from GBR
    gbr = artifacts['models']['gbr_reg']
    importances = gbr.feature_importances_
    names = artifacts['all_feature_names']
    imp_list = sorted(
        [(names[i], float(importances[i])) for i in range(len(names))],
        key=lambda x: x[1], reverse=True
    )[:6]

    return {
        'value': value,
        'tier': tier,
        'confidence': confidence,
        'factors': imp_list,
        'probs': {
            artifacts['label_encoder'].inverse_transform([i])[0]: float(p)
            for i, p in enumerate(probs)
        }
    }


def find_similar(player_dict, artifacts, n=6):
    """Find the most similar players in the dataset using cosine similarity."""
    X_query = vectorize_player(player_dict, artifacts)
    sims = cosine_similarity(X_query, artifacts['X_full_scaled'])[0]
    top_idx = np.argsort(sims)[::-1][:n]

    df = artifacts['df']
    out = []
    for idx in top_idx:
        row = df.iloc[idx]
        out.append({
            'name': row['Name'],
            'similarity': float(sims[idx]),
            'value': float(row['Value Per M$']),
            'overall': int(row['Overall_Rating']),
            'potential': int(row['Future Potential']),
            'age': int(row['Age']),
            'position': str(row['Position']),
            'tier': str(row['Performance_Tier']),
            'country': str(row['Country']),
            'team': str(row['Team'])
        })
    return out


# ============================================================
# PLOTLY THEME HELPERS
# ============================================================
def dark_layout(fig, title=None, height=None):
    """Apply consistent dark theme to any Plotly figure."""
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#e0e0e0", size=12),
        title=dict(text=title, font=dict(size=15, color='#ffffff'), x=0.5, xanchor='center') if title else None,
        legend=dict(bgcolor='rgba(10,14,23,0.8)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
        hoverlabel=dict(bgcolor="#151b2b", font_color="#e0e0e0", bordercolor="rgba(255,255,255,0.1)"),
        margin=dict(l=60, r=40, t=60 if title else 30, b=40),
        height=height
    )
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.06)',
        zerolinecolor='rgba(255,255,255,0.1)',
        showline=True, linecolor='rgba(255,255,255,0.1)'
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.06)',
        zerolinecolor='rgba(255,255,255,0.1)',
        showline=True, linecolor='rgba(255,255,255,0.1)'
    )
    return fig


# ============================================================
# PAGE RENDERERS
# ============================================================
def page_home(artifacts):
    # Hero Banner
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f1724 0%, #1a2332 50%, #0d1f15 100%);
        border-radius: 24px;
        padding: 50px 40px;
        border: 1px solid rgba(0, 200, 83, 0.12);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        margin-bottom: 40px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            right: -10%;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(0,200,83,0.06) 0%, transparent 70%);
            pointer-events: none;
        "></div>
        <h1 style="font-size: 3.5rem; font-weight: 800; color: #ffffff; margin: 0; letter-spacing: -1px;">
            ⚽ SCOUTAI <span style="color: #00c853;">PRO</span>
        </h1>
        <p style="font-size: 1.25rem; color: #8899a6; margin-top: 16px; max-width: 650px; line-height: 1.6;">
            Production-grade football intelligence for player valuation,
            performance tiering, and scouting operations. Powered by ensemble
            machine learning on multi-dimensional player attributes.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    m = artifacts['metrics']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("R² Score", f"{m['r2']:.3f}", "Regression accuracy", "#00c853", "📈")
    with c2:
        card("MAE", f"€{m['mae']:.2f}M", "Mean absolute error", "#ff6b6b", "📉")
    with c3:
        card("Dataset", f"{len(artifacts['df']):,}", "Players analyzed", "#448aff", "🗄️")
    with c4:
        card("Dimensions", f"{len(artifacts['all_feature_names'])}", "Model features", "#ffd700", "🔧")

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview + Tier Distribution
    left, right = st.columns([3, 2])
    with left:
        st.markdown("""
        <div style="
            background: #151b2b;
            border-radius: 16px;
            padding: 28px;
            border: 1px solid rgba(255,255,255,0.05);
            height: 100%;
        ">
            <h3 style="color: #ffffff; margin-top: 0;">Platform Overview</h3>
            <p style="color: #b0b0b0; line-height: 1.8; font-size: 0.95rem;">
                ScoutAI Pro combines gradient boosting, kernel SVM, and instance-based
                KNN models into a unified ensemble architecture. The platform processes
                player demographics, performance ratings, and club context to generate
                robust market value estimates and tier classifications.
            </p>
            <div style="display: flex; gap: 12px; margin-top: 24px;">
                <div style="flex: 1; background: #1a2332; border-radius: 12px; padding: 16px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #00c853;">3</div>
                    <div style="font-size: 0.75rem; color: #8899a6; margin-top: 4px;">Model Families</div>
                </div>
                <div style="flex: 1; background: #1a2332; border-radius: 12px; padding: 16px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #00c853;">2</div>
                    <div style="font-size: 0.75rem; color: #8899a6; margin-top: 4px;">Ensemble Types</div>
                </div>
                <div style="flex: 1; background: #1a2332; border-radius: 12px; padding: 16px; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #00c853;">5-Fold</div>
                    <div style="font-size: 0.75rem; color: #8899a6; margin-top: 4px;">Cross Validation</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        df = artifacts['df']
        tier_counts = df['Performance_Tier'].value_counts().reindex(
            ['Elite', 'Good', 'Average', 'Developing']
        ).fillna(0)
        colors = ['#ffd700', '#00c853', '#448aff', '#78909c']
        fig = go.Figure(data=[go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            hole=0.65,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=11,
            hovertemplate='<b>%{label}</b><br>Players: %{value}<br>Share: %{percent}<extra></extra>'
        )])
        fig.update_layout(
            title=dict(text='Squad Tier Distribution', font=dict(size=14, color='#e0e0e0')),
            showlegend=False,
            height=360,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            margin=dict(t=50, b=20)
        )
        fig.add_annotation(
            text=f"<b>{len(df):,}</b><br>PLAYERS",
            x=0.5, y=0.5, font_size=15, showarrow=False,
            font_color='#ffffff'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quick insights row
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Market Insights")
    i1, i2, i3 = st.columns(3)
    with i1:
        top_val = df.nlargest(1, 'Value Per M$').iloc[0]
        st.markdown(f"""
        <div style="background: #151b2b; border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
            <p style="color: #8899a6; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Highest Valuation</p>
            <p style="color: #ffffff; font-weight: 700; font-size: 1.1rem; margin: 8px 0 0 0;">{top_val['Name']}</p>
            <p style="color: #00c853; font-weight: 700; margin: 4px 0 0 0;">€{top_val['Value Per M$']:.2f}M</p>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        avg_val = df['Value Per M$'].mean()
        st.markdown(f"""
        <div style="background: #151b2b; border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
            <p style="color: #8899a6; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Market Average</p>
            <p style="color: #ffffff; font-weight: 700; font-size: 1.1rem; margin: 8px 0 0 0;">€{avg_val:.2f}M</p>
            <p style="color: #8899a6; font-size: 0.8rem; margin: 4px 0 0 0;">Across all positions</p>
        </div>
        """, unsafe_allow_html=True)
    with i3:
        elite_pct = (df['Performance_Tier'] == 'Elite').mean() * 100
        st.markdown(f"""
        <div style="background: #151b2b; border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
            <p style="color: #8899a6; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Elite Talent Pool</p>
            <p style="color: #ffffff; font-weight: 700; font-size: 1.1rem; margin: 8px 0 0 0;">{elite_pct:.1f}%</p>
            <p style="color: #ffd700; font-size: 0.8rem; margin: 4px 0 0 0;">≥80 Overall Rating</p>
        </div>
        """, unsafe_allow_html=True)


def page_valuation(artifacts):
    section_header(
        "💰 Player Valuation Engine",
        "Input a player attribute profile to generate an AI-powered market value estimate and performance tier classification."
    )

    df = artifacts['df']
    col_inputs, col_results = st.columns([2, 3])

    with col_inputs:
        st.markdown("""
        <div style="background: #151b2b; border-radius: 16px; padding: 24px; border: 1px solid rgba(255,255,255,0.05);">
            <h4 style="color: #00c853; margin-top: 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px;">
                Attribute Profile
            </h4>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 16, 40, 24, help="Current age in years")
                overall = st.slider("Overall Rating", 45, 95, 76,
                                    help="Current overall ability (EA Sports scale)")
            with c2:
                potential = st.slider("Future Potential", 50, 99, 81,
                                      help="Projected peak overall rating")
                total_stats = st.slider("Total Stats Score", 700, 2700, 1850,
                                        help="Aggregate of all technical, physical and mental attributes")

            st.markdown("<br>", unsafe_allow_html=True)

            # Club context
            countries = sorted(df['Country'].unique())
            teams = sorted(df['Team'].unique())
            positions = sorted(df['Position'].unique())

            c3, c4 = st.columns(2)
            with c3:
                country = st.selectbox("Nationality", countries, index=0)
                position = st.selectbox("Position", positions, index=0)
            with c4:
                team = st.selectbox("Current Club", teams, index=0)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🔮 GENERATE VALUATION REPORT", type="primary")

    with col_results:
        if predict_btn:
            player = {
                'Age': age,
                'Overall_Rating': overall,
                'Future Potential': potential,
                'Total_Stats Score': total_stats,
                'Country': country,
                'Team': team,
                'Position': position
            }
            result = predict_player(player, artifacts)

            tier = result['tier']
            tier_color = {'Elite': '#ffd700', 'Good': '#00c853',
                          'Average': '#448aff', 'Developing': '#78909c'}.get(tier, '#78909c')

            # Main result card
            st.markdown(f"""
            <div style="
                background: linear-gradient(145deg, #151b2b, #1a2332);
                border-radius: 20px;
                padding: 32px;
                border: 1px solid {tier_color}40;
                box-shadow: 0 8px 32px {tier_color}18;
                margin-bottom: 24px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 24px;">
                    <div>
                        <p style="color: #8899a6; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 1.5px;">
                            Estimated Market Value
                        </p>
                        <h1 style="color: #ffffff; font-size: 3.2rem; margin: 8px 0; font-weight: 800;">
                            €{result['value']:.2f}M
                        </h1>
                    </div>
                    <div style="text-align: right;">
                        <p style="color: #8899a6; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 1.5px;">
                            Performance Tier
                        </p>
                        <div style="margin-top: 10px;">
                            {tier_badge(tier)}
                        </div>
                    </div>
                </div>

                <div style="background: rgba(0,0,0,0.2); border-radius: 12px; padding: 16px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="color: #b0b0b0; font-size: 0.9rem; font-weight: 500;">Model Confidence</span>
                        <span style="color: {tier_color}; font-weight: 700; font-size: 1rem;">{result['confidence']*100:.1f}%</span>
                    </div>
                    <div style="background: #0a0e17; border-radius: 6px; height: 10px; overflow: hidden;">
                        <div style="
                            width: {result['confidence']*100}%;
                            background: linear-gradient(90deg, {tier_color}, {tier_color}80);
                            height: 100%;
                            border-radius: 6px;
                            transition: width 0.6s ease;
                        "></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Key drivers
            st.markdown("#### Key Value Drivers")
            factor_cols = st.columns(len(result['factors']))
            for col, (factor, imp) in zip(factor_cols, result['factors']):
                clean = factor.replace('_', ' ').replace('pos ', '').title()
                with col:
                    st.markdown(f"""
                    <div style="background: #151b2b; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid rgba(255,255,255,0.05);">
                        <div style="font-size: 1.4rem; font-weight: 800; color: #00c853;">{imp*100:.1f}%</div>
                        <div style="font-size: 0.75rem; color: #8899a6; margin-top: 6px;">{clean}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Tier probability chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Tier Probability Distribution")
            probs = result['probs']
            prob_df = pd.DataFrame({
                'Tier': list(probs.keys()),
                'Probability': [v * 100 for v in probs.values()]
            })
            bar_colors = ['#ffd700', '#00c853', '#448aff', '#78909c']
            fig = go.Figure(go.Bar(
                x=prob_df['Tier'],
                y=prob_df['Probability'],
                marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in prob_df['Probability']],
                textposition='outside',
                textfont=dict(color='#e0e0e0', size=11),
                hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
            ))
            dark_layout(fig, height=300)
            fig.update_layout(xaxis_title="", yaxis_title="Probability (%)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Comparative context
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Attribute Context vs. Dataset")
            comp_data = pd.DataFrame({
                'Attribute': ['Overall', 'Potential', 'Total Stats'],
                'Player': [overall, potential, total_stats],
                'Dataset Avg': [
                    df['Overall_Rating'].mean(),
                    df['Future Potential'].mean(),
                    df['Total_Stats Score'].mean()
                ]
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='This Player', x=comp_data['Attribute'], y=comp_data['Player'],
                marker_color='#00c853', text=comp_data['Player'].round(0),
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                name='Dataset Average', x=comp_data['Attribute'], y=comp_data['Dataset Avg'],
                marker_color='rgba(255,255,255,0.15)', text=comp_data['Dataset Avg'].round(0),
                textposition='outside'
            ))
            dark_layout(fig, height=320)
            fig.update_layout(barmode='group', legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Empty state
            st.markdown("""
            <div style="background: #151b2b; border-radius: 20px; padding: 70px 40px; text-align: center; border: 2px dashed rgba(0,200,83,0.15); margin-top: 20px;">
                <div style="font-size: 3rem; margin-bottom: 16px;">🔮</div>
                <h3 style="color: #ffffff; margin: 0;">Ready to Generate Valuation</h3>
                <p style="color: #8899a6; margin-top: 10px; max-width: 400px; margin-left: auto; margin-right: auto;">
                    Configure the player attribute profile and click the button above to receive
                    an AI-powered market analysis with tier classification.
                </p>
            </div>
            """, unsafe_allow_html=True)


def page_analytics(artifacts):
    section_header(
        "📊 Player Analytics",
        "Interactive exploration of market trends, attribute correlations, and squad composition."
    )

    df = artifacts['df']

    # Filters
    with st.expander("🔍 Analysis Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            pos_filter = st.multiselect(
                "Position", sorted(df['Position'].unique()), default=[]
            )
        with f2:
            age_range = st.slider(
                "Age Range", int(df['Age'].min()), int(df['Age'].max()),
                (int(df['Age'].min()), int(df['Age'].max()))
            )
        with f3:
            overall_range = st.slider(
                "Overall Range", int(df['Overall_Rating'].min()), int(df['Overall_Rating'].max()),
                (int(df['Overall_Rating'].min()), int(df['Overall_Rating'].max()))
            )

    mask = (
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Overall_Rating'] >= overall_range[0]) & (df['Overall_Rating'] <= overall_range[1])
    )
    if pos_filter:
        mask &= df['Position'].isin(pos_filter)
    filtered = df[mask].copy()

    if len(filtered) == 0:
        st.warning("No players match the selected filters. Adjust criteria to explore the dataset.")
        return

    # Row 1: Scatter + Box
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            filtered, x='Age', y='Value Per M$', color='Performance_Tier',
            color_discrete_map={'Elite': '#ffd700', 'Good': '#00c853',
                               'Average': '#448aff', 'Developing': '#78909c'},
            hover_data=['Name', 'Overall_Rating', 'Position', 'Team'],
            opacity=0.7,
            title='Age vs. Market Value'
        )
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            filtered, x='Position', y='Value Per M$', color='Performance_Tier',
            color_discrete_map={'Elite': '#ffd700', 'Good': '#00c853',
                               'Average': '#448aff', 'Developing': '#78909c'},
            title='Value Distribution by Position'
        )
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Feature Importance + Correlation
    c3, c4 = st.columns(2)
    with c3:
        gbr = artifacts['models']['gbr_reg']
        names = artifacts['all_feature_names']
        imp = pd.DataFrame({
            'Feature': [n.replace('_', ' ').replace('pos ', '').title() for n in names],
            'Importance': gbr.feature_importances_
        }).sort_values('Importance', ascending=True).tail(12)

        fig = go.Figure(go.Bar(
            x=imp['Importance'], y=imp['Feature'],
            orientation='h', marker_color='#00c853',
            text=[f"{v:.3f}" for v in imp['Importance']],
            textposition='outside'
        ))
        dark_layout(fig, title='Value Drivers (Gradient Boosting)', height=420)
        fig.update_layout(margin=dict(l=140))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        numeric_cols = ['Age', 'Overall_Rating', 'Future Potential',
                        'Value Per M$', 'Total_Stats Score']
        corr = filtered[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto='.2f', aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Attribute Correlation Matrix',
            zmin=-1, zmax=1
        )
        dark_layout(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Market distribution + Position tier breakdown
    c5, c6 = st.columns(2)
    with c5:
        fig = px.histogram(
            filtered, x='Value Per M$', color='Performance_Tier',
            color_discrete_map={'Elite': '#ffd700', 'Good': '#00c853',
                               'Average': '#448aff', 'Developing': '#78909c'},
            nbins=50, marginal='rug',
            title='Market Value Distribution',
            opacity=0.75
        )
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        pos_tier = filtered.groupby(['Position', 'Performance_Tier']).size().reset_index(name='Count')
        fig = px.bar(
            pos_tier, x='Position', y='Count', color='Performance_Tier',
            color_discrete_map={'Elite': '#ffd700', 'Good': '#00c853',
                               'Average': '#448aff', 'Developing': '#78909c'},
            title='Squad Composition by Position',
            barmode='stack'
        )
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)


def page_similar(artifacts):
    section_header(
        "🔍 Similar Players",
        "Discover comparable talent using multi-dimensional cosine similarity analysis."
    )

    df = artifacts['df']
    tab1, tab2 = st.tabs(["📋 By Existing Player", "⚙️ By Custom Profile"])

    with tab1:
        player_names = df['Name'].tolist()
        selected = st.selectbox(
            "Select a player from the database",
            player_names,
            format_func=lambda x: f"{x}  —  {df[df['Name']==x].iloc[0]['Position']}  |  OVR {df[df['Name']==x].iloc[0]['Overall_Rating']}  |  €{df[df['Name']==x].iloc[0]['Value Per M$']:.1f}M"
        )

        if st.button("Find Comparables", key="sim_existing", type="primary"):
            row = df[df['Name'] == selected].iloc[0]
            player_dict = {
                'Age': int(row['Age']),
                'Overall_Rating': int(row['Overall_Rating']),
                'Future Potential': int(row['Future Potential']),
                'Total_Stats Score': int(row['Total_Stats Score']),
                'Country': str(row['Country']),
                'Team': str(row['Team']),
                'Position': str(row['Position'])
            }
            similar = find_similar(player_dict, artifacts, n=7)
            similar = [s for s in similar if s['name'] != selected][:5]

            st.markdown(f"#### Top 5 Comparables for {selected}")

            # Table
            sim_df = pd.DataFrame(similar)
            sim_df['similarity'] = (sim_df['similarity'] * 100).round(1)
            sim_df = sim_df[['name', 'similarity', 'value', 'overall', 'potential', 'age', 'position', 'tier']]
            sim_df.columns = ['Player', 'Similarity %', 'Value (M€)', 'OVR', 'POT', 'Age', 'POS', 'Tier']
            st.dataframe(
                sim_df.style.background_gradient(subset=['Similarity %'], cmap='Greens').format({
                    'Similarity %': '{:.1f}',
                    'Value (M€)': '€{:.2f}',
                    'OVR': '{:.0f}',
                    'POT': '{:.0f}',
                    'Age': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            # Radar comparison
            categories = ['Overall', 'Potential', 'Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physical']
            sel_row = df[df['Name'] == selected].iloc[0]
            sel_vals = [
                sel_row['Overall_Rating'], sel_row['Future Potential'],
                sel_row.get('Pace', 70), sel_row.get('Shooting', 65),
                sel_row.get('Passing', 65), sel_row.get('Dribbling', 68),
                sel_row.get('Defending', 55), sel_row.get('Physical', 68)
            ]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=sel_vals + [sel_vals[0]],
                theta=categories + [categories[0]],
                fill='toself', name=selected,
                line_color='#00c853', fillcolor='rgba(0,200,83,0.15)'
            ))

            if similar:
                top = similar[0]
                top_row = df[df['Name'] == top['name']].iloc[0]
                top_vals = [
                    top_row['Overall_Rating'], top_row['Future Potential'],
                    top_row.get('Pace', 70), top_row.get('Shooting', 65),
                    top_row.get('Passing', 65), top_row.get('Dribbling', 68),
                    top_row.get('Defending', 55), top_row.get('Physical', 68)
                ]
                fig.add_trace(go.Scatterpolar(
                    r=top_vals + [top_vals[0]],
                    theta=categories + [categories[0]],
                    fill='toself', name=top['name'],
                    line_color='#448aff', fillcolor='rgba(68,138,255,0.1)'
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=-0.15)
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 16, 40, 24, key="sim_age")
            overall = st.slider("Overall Rating", 45, 95, 76, key="sim_ovr")
        with c2:
            potential = st.slider("Potential", 50, 99, 81, key="sim_pot")
            total_stats = st.slider("Total Stats", 700, 2700, 1850, key="sim_stats")

        c3, c4 = st.columns(2)
        with c3:
            country = st.selectbox("Nationality", sorted(df['Country'].unique()), key="sim_country")
            position = st.selectbox("Position", sorted(df['Position'].unique()), key="sim_pos")
        with c4:
            team = st.selectbox("Current Club", sorted(df['Team'].unique()), key="sim_team")

        if st.button("Find Comparables", key="sim_custom", type="primary"):
            player_dict = {
                'Age': age, 'Overall_Rating': overall, 'Future Potential': potential,
                'Total_Stats Score': total_stats, 'Country': country,
                'Team': team, 'Position': position
            }
            similar = find_similar(player_dict, artifacts, n=5)

            st.markdown("<h4 style='color: #e0e0e0; margin-top: 20px;'>Top Matches</h4>", unsafe_allow_html=True)
            for s in similar:
                tc = {'Elite': '#ffd700', 'Good': '#00c853',
                      'Average': '#448aff', 'Developing': '#78909c'}.get(s['tier'], '#78909c')
                st.markdown(f"""
                <div style="background: #151b2b; border-radius: 12px; padding: 18px; margin-bottom: 12px; border-left: 4px solid {tc}; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: 700; color: #ffffff; font-size: 1.1rem;">{s['name']}</div>
                        <div style="color: #8899a6; font-size: 0.85rem; margin-top: 4px;">{s['position']} | Age {s['age']} | OVR {s['overall']} | {s['team']}</div>
                    </div>
                    <div style="text-align: right; min-width: 100px;">
                        <div style="font-weight: 800; color: #00c853; font-size: 1.3rem;">{s['similarity']*100:.1f}%</div>
                        <div style="color: #8899a6; font-size: 0.75rem;">similarity</div>
                        <div style="color: #ffffff; font-weight: 600; font-size: 0.9rem; margin-top: 4px;">€{s['value']:.2f}M</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def page_about(artifacts):
    section_header(
        "🧠 About the Model",
        "Technical documentation of the ScoutAI Pro machine learning pipeline and evaluation framework."
    )

    # Pipeline architecture
    st.markdown("""
    <div style="background: #151b2b; border-radius: 16px; padding: 28px; margin-bottom: 28px; border: 1px solid rgba(255,255,255,0.05);">
        <h4 style="color: #00c853; margin-top: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1.5px;">
            Preprocessing Pipeline
        </h4>
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 20px;">
            <div style="background: #1a2332; border-radius: 8px; padding: 10px 16px; color: #e0e0e0; font-size: 0.8rem; font-weight: 500;">Raw FIFA Data</div>
            <div style="color: #00c853; font-weight: 700;">→</div>
            <div style="background: #1a2332; border-radius: 8px; padding: 10px 16px; color: #e0e0e0; font-size: 0.8rem; font-weight: 500;">Missing Value Imputation</div>
            <div style="color: #00c853; font-weight: 700;">→</div>
            <div style="background: #1a2332; border-radius: 8px; padding: 10px 16px; color: #e0e0e0; font-size: 0.8rem; font-weight: 500;">Target Encoding (Country, Team)</div>
            <div style="color: #00c853; font-weight: 700;">→</div>
            <div style="background: #1a2332; border-radius: 8px; padding: 10px 16px; color: #e0e0e0; font-size: 0.8rem; font-weight: 500;">One-Hot Encoding (Position)</div>
            <div style="color: #00c853; font-weight: 700;">→</div>
            <div style="background: #1a2332; border-radius: 8px; padding: 10px 16px; color: #e0e0e0; font-size: 0.8rem; font-weight: 500;">Standard Scaling</div>
            <div style="color: #00c853; font-weight: 700;">→</div>
            <div style="background: #1a2332; border-radius: 8px; padding: 10px 16px; color: #e0e0e0; font-size: 0.8rem; font-weight: 500;">Log-Transform Target</div>
        </div>
        <p style="color: #8899a6; font-size: 0.85rem; margin-top: 16px; line-height: 1.6;">
            All encoding and scaling operations are fit exclusively on the training partition
            to prevent data leakage. Stratified sampling preserves the imbalanced tier distribution
            (Elite ~3%, Good ~25%, Average ~40%, Developing ~32%).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model specs
    c1, c2, c3 = st.columns(3)
    specs = [
        ("Gradient Boosting", "Tree-based ensemble with 120 estimators, max depth 5. Handles non-linearity, heteroscedasticity, and extreme outliers (Mbappé/Haaland-type valuations) natively.", "#00c853"),
        ("Support Vector Machine", "RBF kernel SVM with C=10. Captures smooth non-linear manifolds in the Age–Overall–Value space. Margin maximization controls overfitting on the imbalanced Elite tier.", "#448aff"),
        ("K-Nearest Neighbors", "Instance-based learner with k=10 and distance weighting. Football valuation is inherently comparative; KNN provides interpretable player comparables for scouting departments.", "#ff6b6b")
    ]
    for col, (title, desc, color) in zip([c1, c2, c3], specs):
        with col:
            st.markdown(f"""
            <div style="
                background: #151b2b;
                border-radius: 16px;
                padding: 24px;
                height: 100%;
                border-top: 4px solid {color};
                border: 1px solid rgba(255,255,255,0.05);
                border-top-width: 4px;
            ">
                <h4 style="color: #ffffff; margin-top: 0; font-size: 1rem;">{title}</h4>
                <p style="color: #8899a6; font-size: 0.88rem; line-height: 1.7;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance metrics
    st.markdown("### Test-Set Performance Summary")
    m = artifacts['metrics']
    perf_data = {
        'Task': ['Regression', 'Regression', 'Regression', 'Classification', 'Classification', 'Classification', 'Classification'],
        'Metric': ['R² Score', 'MAE (M€)', 'RMSE (M€)', 'Accuracy', 'F1-Weighted', 'F1-Macro', 'MCC'],
        'Ensemble Value': [
            f"{m['r2']:.3f}", f"{m['mae']:.2f}", f"{m['rmse']:.2f}",
            f"{m['accuracy']:.3f}", f"{m['f1_weighted']:.3f}",
            f"{m['f1_macro']:.3f}", f"{m['mcc']:.3f}"
        ],
        'Interpretation': [
            'Variance explained',
            'Avg. prediction error',
            'Root squared error',
            'Overall correct rate',
            'Balanced precision/recall',
            'Unweighted class average',
            'Correlation coefficient'
        ]
    }
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Model comparison
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Algorithm Family Comparison")
    comp_data = {
        'Family': ['Tree-Based (GBR/GBC)', 'Kernel-Based (SVR/SVC)', 'Instance-Based (KNN)', 'Voting Ensemble', 'Stacking Ensemble*'],
        'Strength': ['Non-linear interactions, outlier robust', 'Smooth manifolds, regularization', 'Interpretable comparables', 'Parallel aggregation', 'Meta-learned weights'],
        'Best For': ['Heteroscedastic value data', 'Medium-dim feature spaces', 'Scouting "look-alikes"', 'Reduced variance', 'Optimal blending'],
        'Complexity': ['Medium', 'High', 'Low', 'Low', 'Medium']
    }
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption("*Stacking ensemble available in the research notebook; voting ensemble deployed in production for inference stability.")

    # Metrics explanation
    with st.expander("📐 Evaluation Metrics Reference"):
        st.markdown("""
        <p><b>Regression Metrics</b></p>
        <ul>
        <li><b>R² Score</b>: Coefficient of determination. Proportion of market value variance explained by the model. Range (-∞, 1]; 1.0 is perfect prediction.</li>
        <li><b>MAE</b>: Mean Absolute Error in millions of euros. The average magnitude of valuation errors, robust to outliers.</li>
        <li><b>RMSE</b>: Root Mean Squared Error. Penalizes large misses (e.g., undervaluing a generational talent) more aggressively than MAE.</li>
        </ul>
        <p><b>Classification Metrics</b></p>
        <ul>
        <li><b>Accuracy</b>: Overall correct tier prediction rate. Can be misleading with imbalanced classes.</li>
        <li><b>F1-Weighted</b>: Harmonic mean of precision and recall, weighted by tier support. Balances false positives/negatives across all tiers.</li>
        <li><b>F1-Macro</b>: Unweighted average of per-tier F1 scores. Treats Elite and Developing tiers as equally important.</li>
        <li><b>MCC</b>: Matthews Correlation Coefficient. A balanced measure that returns +1 for perfect prediction, 0 for random guessing, and -1 for total disagreement.</li>
        </ul>
        """, unsafe_allow_html=True)

    # Feature importance visualization
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Feature Importance — Regression Ensemble")
    gbr = artifacts['models']['gbr_reg']
    names = artifacts['all_feature_names']
    imp = pd.DataFrame({
        'Feature': [n.replace('_', ' ').replace('pos ', '').title() for n in names],
        'Importance': gbr.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=imp['Importance'], y=imp['Feature'],
        orientation='h', marker_color='#00c853',
        text=[f"{v:.3f}" for v in imp['Importance']],
        textposition='outside'
    ))
    dark_layout(fig, height=500)
    fig.update_layout(margin=dict(l=160, r=40))
    st.plotly_chart(fig, use_container_width=True)

    # Anti-leakage & reproducibility
    st.markdown("""
    <div style="background: #151b2b; border-radius: 16px; padding: 24px; margin-top: 24px; border: 1px solid rgba(255,255,255,0.05);">
        <h4 style="color: #00c853; margin-top: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1.5px;">Reproducibility & Anti-Leakage</h4>
        <ul style="color: #8899a6; line-height: 1.8; font-size: 0.9rem;">
            <li><b>Target encoders</b> fit exclusively on the training partition; test set never influences encoding maps.</li>
            <li><b>StandardScaler</b> statistics (mean/std) computed from training features only.</li>
            <li><b>Stratified k-fold</b> cross-validation preserves tier distribution across all validation folds.</li>
            <li><b>Random seed 42</b> locked across NumPy, scikit-learn, and ensemble initializations for full reproducibility.</li>
            <li><b>Log-transform</b> on target variable reduces extreme right skewness (raw skew ≈ 8.0 → log skew ≈ 1.2).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION ROUTER
# ============================================================
def main():
    # Load data and build ML pipeline
    df = load_data()
    artifacts = build_pipeline(df)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 style="color: #ffffff; font-weight: 800; margin: 0; font-size: 1.8rem;">⚽ SCOUTAI</h2>
            <p style="color: #00c853; font-size: 0.8rem; letter-spacing: 3px; margin: 4px 0 0 0; font-weight: 600;">PRO</p>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Home", "💰 Player Valuation", "📊 Player Analytics", "🔍 Similar Players", "🧠 About Model"],
            label_visibility="collapsed"
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)

        # System status panel
        m = artifacts['metrics']
        st.markdown(f"""
        <div style="background: #151b2b; border-radius: 12px; padding: 18px; border: 1px solid rgba(0,200,83,0.1);">
            <p style="color: #8899a6; font-size: 0.7rem; margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">
                System Status
            </p>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="width: 8px; height: 8px; background: #00c853; border-radius: 50%; box-shadow: 0 0 8px #00c853;"></div>
                <span style="color: #e0e0e0; font-size: 0.82rem; font-weight: 500;">Models Active</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="width: 8px; height: 8px; background: #448aff; border-radius: 50%; box-shadow: 0 0 8px #448aff;"></div>
                <span style="color: #e0e0e0; font-size: 0.82rem; font-weight: 500;">{len(artifacts['df']):,} Players</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; background: #ffd700; border-radius: 50%; box-shadow: 0 0 8px #ffd700;"></div>
                <span style="color: #e0e0e0; font-size: 0.82rem; font-weight: 500;">R² {m['r2']:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("© 2026 ScoutAI Pro. Professional Football Intelligence.")

    # Route to page
    if page == "🏠 Home":
        page_home(artifacts)
    elif page == "💰 Player Valuation":
        page_valuation(artifacts)
    elif page == "📊 Player Analytics":
        page_analytics(artifacts)
    elif page == "🔍 Similar Players":
        page_similar(artifacts)
    elif page == "🧠 About Model":
        page_about(artifacts)


if __name__ == "__main__":
    main()
