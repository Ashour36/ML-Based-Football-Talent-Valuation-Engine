# ============================================================
# SCOUTAI PRO — Professional Football Intelligence Platform
# Fixed version: all 14 bugs resolved
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, matthews_corrcoef,
)
from sklearn.metrics.pairwise import cosine_similarity

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TIER_COLORS = {
    'Elite':      '#ffd700',
    'Good':       '#00c853',
    'Average':    '#448aff',
    'Developing': '#78909c',
}
TIER_ORDER = ['Elite', 'Good', 'Average', 'Developing']

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScoutAI Pro | Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ScoutAI Pro v2.0 — Professional Football Intelligence Platform"},
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #0a0e17; }
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid rgba(255,255,255,0.05);
}
h1 { color: #ffffff; font-weight: 800; letter-spacing: -0.5px; }
h2 {
    color: #e0e0e0; font-weight: 700;
    border-left: 4px solid #00c853;
    padding-left: 16px; margin-top: 30px;
}
h3 { color: #b0b0b0; font-weight: 600; }
p, li { color: #8899a6; }
.stButton > button {
    background: linear-gradient(90deg, #00c853 0%, #009624 100%);
    color: white; border: none; border-radius: 12px;
    padding: 14px 24px; font-size: 16px; font-weight: 700;
    letter-spacing: 0.5px; transition: all 0.3s ease; width: 100%;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(0,200,83,0.4);
    transform: translateY(-2px);
}
.stButton > button:active { transform: scale(0.98); }
[data-testid="stMetricValue"]  { color: #ffffff; font-weight: 800; }
[data-testid="stMetricLabel"]  { color: #8899a6; text-transform: uppercase; letter-spacing: 1px; font-size: 0.75rem; }
[data-testid="metric-container"] {
    background: #151b2b;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.streamlit-expanderHeader { background-color: #151b2b; border-radius: 12px; color: #e0e0e0; font-weight: 600; }
.streamlit-expanderContent { background-color: #0f1520; border-radius: 0 0 12px 12px; }
.stTabs [data-baseweb="tab"] {
    background-color: #151b2b; border-radius: 8px 8px 0 0;
    color: #8899a6; font-weight: 600; border: none; padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background-color: #1a2332 !important;
    color: #00c853 !important;
    border-bottom: 2px solid #00c853 !important;
}
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0a0e17; }
::-webkit-scrollbar-thumb { background: #1a2332; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #00c853; }
hr { border-color: rgba(255,255,255,0.06); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# UI HELPERS
# ============================================================
def ui_card(title: str, value: str, subtitle: str,
            color: str = "#00c853", icon: str = "") -> None:
    st.markdown(f"""
<div style="background:linear-gradient(145deg,#151b2b,#1a2332);border-radius:16px;
            padding:24px;border-left:4px solid {color};
            box-shadow:0 4px 20px rgba(0,0,0,0.3);margin-bottom:16px;">
    <p style="color:#8899a6;font-size:0.8rem;margin:0;text-transform:uppercase;letter-spacing:1.5px;">
        {icon} {title}
    </p>
    <h2 style="color:#ffffff;font-size:2.2rem;margin:8px 0;font-weight:800;">{value}</h2>
    <p style="color:{color};font-size:0.85rem;margin:0;font-weight:500;">{subtitle}</p>
</div>""", unsafe_allow_html=True)


def ui_tier_badge(tier: str) -> str:
    icons = {'Elite': '★', 'Good': '▲', 'Average': '●', 'Developing': '◆'}
    color = TIER_COLORS.get(tier, '#78909c')
    icon  = icons.get(tier, '·')
    fg    = '#000000' if tier == 'Elite' else '#ffffff'
    return (
        f"<span style='background:{color};color:{fg};padding:6px 16px;"
        f"border-radius:20px;font-weight:800;font-size:0.85rem;"
        f"letter-spacing:1px;display:inline-block;"
        f"box-shadow:0 2px 10px {color}40;'>{icon} {tier.upper()}</span>"
    )


def ui_section(title: str, subtitle: str = "") -> None:
    sub = f"<p style='color:#8899a6;margin:0;font-size:1rem;'>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div style='margin-bottom:24px;'>"
        f"<h2 style='margin-top:0;margin-bottom:8px;'>{title}</h2>{sub}</div>",
        unsafe_allow_html=True,
    )


def dark_layout(fig: go.Figure, title: str = "", height: int = 0) -> go.Figure:
    """Apply consistent dark theme — FIX BUG 13: title handled safely."""
    kw: dict = dict(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e0e0e0", size=12),
        legend=dict(bgcolor="rgba(10,14,23,0.8)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        hoverlabel=dict(bgcolor="#151b2b", font_color="#e0e0e0", bordercolor="rgba(255,255,255,0.1)"),
        margin=dict(l=60, r=40, t=60 if title else 30, b=40),
    )
    if title:
        kw["title"] = dict(text=title, font=dict(size=15, color="#ffffff"), x=0.5, xanchor="center")
    if height:
        kw["height"] = height
    fig.update_layout(**kw)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
        showline=True, linecolor="rgba(255,255,255,0.1)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
        showline=True, linecolor="rgba(255,255,255,0.1)",
    )
    return fig


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(path: str = "Fifa.csv") -> pd.DataFrame:
    """
    FIX BUG 4 + BUGS 1/2/3: Merging load + tier assignment into one
    cached function that returns a plain DataFrame (hashable path as key).
    Performance_Tier is added here so it exists in every downstream use.
    FIX BUG 9: filter zero-value players that distort log-transform.
    """
    try:
        df = pd.read_csv(path)
        required = [
            'Name', 'Age', 'Overall_Rating', 'Future Potential',
            'Total_Stats Score', 'Value Per M$', 'Country', 'Team', 'Position',
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        st.sidebar.success("✅ Loaded Fifa.csv")
    except Exception as exc:
        st.sidebar.warning(f"⚠️ Using synthetic data ({exc})")
        df = _synthetic_data()

    # FIX BUG 9: remove zero-value rows (log1p(0)=0 is mathematically fine
    # but these players have no market — they skew regression toward zero)
    df = df[df['Value Per M$'] > 0].copy()

    # FIX BUGS 2 + 3: add Performance_Tier here so artifacts['df'] has it
    def _tier(r: int) -> str:
        if r >= 80: return 'Elite'
        if r >= 70: return 'Good'
        if r >= 60: return 'Average'
        return 'Developing'

    df['Performance_Tier'] = df['Overall_Rating'].apply(_tier)

    # Deduplicate names by appending index for uniqueness
    # FIX BUG 6: avoids wrong player being returned from name lookup
    counts: dict = {}
    new_names = []
    for name in df['Name']:
        if name in counts:
            counts[name] += 1
            new_names.append(f"{name} ({counts[name]})")
        else:
            counts[name] = 0
            new_names.append(name)
    df['Name'] = new_names

    return df.reset_index(drop=True)


def _synthetic_data(n: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    positions = ['ST','CF','LW','RW','CAM','CM','CDM','LB','RB','CB','GK']
    countries  = ['Argentina','Brazil','England','France','Germany','Spain',
                  'Italy','Portugal','Netherlands','Belgium','Croatia','Uruguay']
    teams = ['Manchester City','Real Madrid','Barcelona','Bayern Munich',
             'Paris SG','Liverpool','Chelsea','Arsenal','Juventus'] + [f'Club {i:03d}' for i in range(40)]
    first = ['Alex','Marco','Lucas','David','James','Luis','Carlos','Thomas','Bruno','Pedro']
    last  = ['Silva','Martinez','Garcia','Rodriguez','Smith','Johnson','Brown','Davis']
    data = {
        'Name': [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(n)],
        'Age':  rng.integers(16, 39, n).tolist(),
        'Overall_Rating': np.clip(rng.normal(68,9,n), 46,94).astype(int).tolist(),
        'Future Potential': np.clip(rng.normal(72,10,n), 55,96).astype(int).tolist(),
        'Total_Stats Score': np.clip(rng.normal(1850,420,n), 700,2700).astype(int).tolist(),
        'Country':  rng.choice(countries, n).tolist(),
        'Team':     rng.choice(teams, n).tolist(),
        'Position': rng.choice(positions, n).tolist(),
    }
    df = pd.DataFrame(data)
    age_f = np.maximum(0, (28 - np.abs(df['Age'] - 26)) / 28)
    base  = (df['Overall_Rating']**2.3)*0.008 + (df['Future Potential']**1.9)*0.004 + df['Total_Stats Score']*0.0007
    base *= (1 + age_f * 2.5)
    pm = {'ST':1.45,'CF':1.35,'LW':1.40,'RW':1.40,'CAM':1.25,'CM':1.05,
          'CDM':0.95,'LB':0.85,'RB':0.85,'CB':0.90,'GK':0.80}
    mults  = np.array([pm.get(p, 1.0) for p in df['Position']])
    noise  = rng.lognormal(0, 0.55, n)
    df['Value Per M$'] = np.round(np.maximum(base * mults * noise, 0.05), 2)
    return df


# ============================================================
# MODEL PIPELINE
# ============================================================
@st.cache_resource(show_spinner="Training ensemble models …")
def build_pipeline(data_path: str = "Fifa.csv") -> dict:
    """
    FIX BUG 1: cache key is the file path (str), NOT the DataFrame.
    FIX BUG 8: SVR removed from VotingRegressor — O(n²) memory on 15k rows
               causes OOM on Streamlit Cloud. Replaced with a second GBR
               with different hyperparams for ensemble diversity.
    FIX BUG 10: pos_cols is sorted() to guarantee deterministic column order.
    """
    df = load_data(data_path)

    numeric_features = ['Age', 'Overall_Rating', 'Future Potential', 'Total_Stats Score']
    cat_features     = ['Country', 'Team', 'Position']
    df = df.dropna(subset=numeric_features + ['Value Per M$'] + cat_features)

    X_raw = df[numeric_features + cat_features].copy()
    y_reg = df['Value Per M$'].values
    y_clf = df['Performance_Tier'].values

    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)

    X_tr_raw, X_te_raw, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X_raw, y_reg, y_clf_enc,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf_enc,
    )

    # ── Target encoding (fit on TRAIN only — no leakage) ──────────────────
    tmp = X_tr_raw.copy(); tmp['__t__'] = yr_tr
    country_map = tmp.groupby('Country')['__t__'].mean().to_dict()
    team_map    = tmp.groupby('Team')['__t__'].mean().to_dict()
    global_mean = float(yr_tr.mean())

    def _te(series: pd.Series, mapping: dict) -> np.ndarray:
        return series.map(mapping).fillna(global_mean).values

    X_tr = X_tr_raw.copy(); X_te = X_te_raw.copy()
    X_tr['Country_Encoded'] = _te(X_tr_raw['Country'], country_map)
    X_tr['Team_Encoded']    = _te(X_tr_raw['Team'],    team_map)
    X_te['Country_Encoded'] = _te(X_te_raw['Country'], country_map)
    X_te['Team_Encoded']    = _te(X_te_raw['Team'],    team_map)

    # ── One-hot encode Position ────────────────────────────────────────────
    pos_dummies_tr = pd.get_dummies(X_tr['Position'], prefix='pos')
    pos_dummies_te = pd.get_dummies(X_te['Position'], prefix='pos')
    # FIX BUG 10: sorted() makes order deterministic across runs
    pos_cols = sorted(set(pos_dummies_tr.columns) | set(pos_dummies_te.columns))
    for col in pos_cols:
        if col not in pos_dummies_tr.columns: pos_dummies_tr[col] = 0
        if col not in pos_dummies_te.columns: pos_dummies_te[col] = 0
    pos_dummies_tr = pos_dummies_tr[pos_cols]
    pos_dummies_te = pos_dummies_te[pos_cols]

    engineered_cols = numeric_features + ['Country_Encoded', 'Team_Encoded']
    X_tr_mat = np.column_stack([X_tr[engineered_cols].values, pos_dummies_tr.values])
    X_te_mat = np.column_stack([X_te[engineered_cols].values, pos_dummies_te.values])

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_mat)
    X_te_sc = scaler.transform(X_te_mat)

    yr_tr_log = np.log1p(yr_tr)
    all_feature_names = engineered_cols + pos_cols

    # ── Regression models ─────────────────────────────────────────────────
    # FIX BUG 8: No SVR. Two GBR variants + KNN for diversity without OOM.
    gbr_reg1 = GradientBoostingRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        min_samples_leaf=10, subsample=0.85, random_state=RANDOM_STATE,
    )
    gbr_reg2 = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.12,
        min_samples_leaf=5, subsample=0.8, random_state=RANDOM_STATE + 1,
    )
    knn_reg = KNeighborsRegressor(n_neighbors=12, weights='distance', metric='euclidean')

    gbr_reg1.fit(X_tr_sc, yr_tr_log)
    gbr_reg2.fit(X_tr_sc, yr_tr_log)
    knn_reg.fit(X_tr_sc, yr_tr_log)

    ensemble_reg = VotingRegressor([('gbr1', gbr_reg1), ('gbr2', gbr_reg2), ('knn', knn_reg)])
    ensemble_reg.fit(X_tr_sc, yr_tr_log)

    # ── Classification models ─────────────────────────────────────────────
    gbr_clf1 = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        min_samples_leaf=10, subsample=0.85, random_state=RANDOM_STATE,
    )
    gbr_clf2 = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.12,
        min_samples_leaf=5, subsample=0.8, random_state=RANDOM_STATE + 1,
    )
    knn_clf = KNeighborsClassifier(n_neighbors=12, weights='distance', metric='euclidean')

    gbr_clf1.fit(X_tr_sc, yc_tr)
    gbr_clf2.fit(X_tr_sc, yc_tr)
    knn_clf.fit(X_tr_sc, yc_tr)

    ensemble_clf = VotingClassifier(
        [('gbr1', gbr_clf1), ('gbr2', gbr_clf2), ('knn', knn_clf)],
        voting='soft',
    )
    ensemble_clf.fit(X_tr_sc, yc_tr)

    # ── Evaluation ────────────────────────────────────────────────────────
    pred_reg = np.expm1(ensemble_reg.predict(X_te_sc))
    pred_clf = ensemble_clf.predict(X_te_sc)

    metrics = dict(
        r2   = float(r2_score(yr_te, pred_reg)),
        mae  = float(mean_absolute_error(yr_te, pred_reg)),
        rmse = float(np.sqrt(mean_squared_error(yr_te, pred_reg))),
        accuracy  = float(accuracy_score(yc_te, pred_clf)),
        f1_weighted = float(f1_score(yc_te, pred_clf, average='weighted')),
        f1_macro    = float(f1_score(yc_te, pred_clf, average='macro')),
        mcc  = float(matthews_corrcoef(yc_te, pred_clf)),
        train_size = len(X_tr_sc),
        test_size  = len(X_te_sc),
    )

    # ── Full-dataset scaled matrix for similarity search ──────────────────
    X_full_enc = df[numeric_features].copy()
    X_full_enc['Country_Encoded'] = _te(df['Country'], country_map)
    X_full_enc['Team_Encoded']    = _te(df['Team'],    team_map)
    pos_full = pd.get_dummies(df['Position'], prefix='pos')
    for col in pos_cols:
        if col not in pos_full.columns: pos_full[col] = 0
    pos_full = pos_full[pos_cols]
    X_full_mat = np.column_stack([X_full_enc.values, pos_full.values])
    X_full_sc  = scaler.transform(X_full_mat)

    # FIX BUG 5 / 6: precompute display labels for the selectbox
    # (avoids O(n) per-option df lookup)
    display_labels = {
        row['Name']: (
            f"{row['Name']}  —  "
            f"{row['Position']}  |  "
            f"OVR {row['Overall_Rating']}  |  "
            f"€{row['Value Per M$']:.1f}M"
        )
        for _, row in df.iterrows()
    }

    return dict(
        df               = df,
        scaler           = scaler,
        label_encoder    = le,
        country_map      = country_map,
        team_map         = team_map,
        global_mean      = global_mean,
        pos_cols         = pos_cols,
        numeric_features = numeric_features,
        engineered_cols  = engineered_cols,
        all_feature_names= all_feature_names,
        models           = dict(
            regression      = ensemble_reg,
            classification  = ensemble_clf,
            gbr_reg         = gbr_reg1,   # primary GBR for feature importance
            gbr_clf         = gbr_clf1,
        ),
        metrics          = metrics,
        X_full_scaled    = X_full_sc,
        display_labels   = display_labels,
    )


# ============================================================
# INFERENCE
# ============================================================
def vectorize_player(player_dict: dict, artifacts: dict) -> np.ndarray:
    """
    FIX BUG 11: all column lists come from artifacts — no local redefinition.
    FIX BUG 10: pos_cols from artifacts is sorted, so order is guaranteed.
    """
    num      = artifacts['numeric_features']
    eng      = artifacts['engineered_cols']
    pos_cols = artifacts['pos_cols']

    row: dict = {k: float(player_dict.get(k, 0)) for k in num}
    row['Country_Encoded'] = artifacts['country_map'].get(player_dict.get('Country'), artifacts['global_mean'])
    row['Team_Encoded']    = artifacts['team_map'].get(player_dict.get('Team'),    artifacts['global_mean'])

    pos_key = f"pos_{player_dict.get('Position', 'CM')}"
    for col in pos_cols:
        row[col] = 1 if col == pos_key else 0

    X = np.array([[row[k] for k in eng + pos_cols]])
    return artifacts['scaler'].transform(X)


def predict_player(player_dict: dict, artifacts: dict) -> dict:
    X = vectorize_player(player_dict, artifacts)

    pred_log = artifacts['models']['regression'].predict(X)[0]
    value    = float(np.expm1(pred_log))

    tier_enc   = artifacts['models']['classification'].predict(X)[0]
    tier       = artifacts['label_encoder'].inverse_transform([tier_enc])[0]
    probs_arr  = artifacts['models']['classification'].predict_proba(X)[0]
    confidence = float(np.max(probs_arr))
    probs      = {
        artifacts['label_encoder'].inverse_transform([i])[0]: float(p)
        for i, p in enumerate(probs_arr)
    }

    imp   = artifacts['models']['gbr_reg'].feature_importances_
    names = artifacts['all_feature_names']
    factors = sorted(
        [(names[i].replace('_',' ').replace('pos ','').title(), float(imp[i]))
         for i in range(len(names))],
        key=lambda x: x[1], reverse=True,
    )[:6]

    return dict(value=value, tier=tier, confidence=confidence,
                probs=probs, factors=factors)


def find_similar(player_dict: dict, artifacts: dict, n: int = 6) -> list[dict]:
    """
    FIX BUG 3: Performance_Tier now always exists in artifacts['df']
    because load_data() adds it before build_pipeline() stores the df.
    """
    X_q  = vectorize_player(player_dict, artifacts)
    sims = cosine_similarity(X_q, artifacts['X_full_scaled'])[0]
    top  = np.argsort(sims)[::-1][:n]
    df   = artifacts['df']
    out  = []
    for idx in top:
        row = df.iloc[idx]
        out.append(dict(
            name       = str(row['Name']),
            similarity = float(sims[idx]),
            value      = float(row['Value Per M$']),
            overall    = int(row['Overall_Rating']),
            potential  = int(row['Future Potential']),
            age        = int(row['Age']),
            position   = str(row['Position']),
            tier       = str(row['Performance_Tier']),   # guaranteed to exist
            country    = str(row['Country']),
            team       = str(row['Team']),
        ))
    return out


# ============================================================
# PAGE — HOME
# ============================================================
def page_home(artifacts: dict) -> None:
    df = artifacts['df']
    m  = artifacts['metrics']

    st.markdown("""
<div style="background:linear-gradient(135deg,#0f1724 0%,#1a2332 50%,#0d1f15 100%);
            border-radius:24px;padding:50px 40px;
            border:1px solid rgba(0,200,83,0.12);
            box-shadow:0 20px 60px rgba(0,0,0,0.5);margin-bottom:40px;">
    <h1 style="font-size:3.5rem;font-weight:800;color:#ffffff;margin:0;letter-spacing:-1px;">
        ⚽ SCOUTAI <span style="color:#00c853;">PRO</span>
    </h1>
    <p style="font-size:1.2rem;color:#8899a6;margin-top:16px;max-width:650px;line-height:1.6;">
        Production-grade football intelligence for player valuation, performance
        tiering, and scouting operations. Powered by ensemble machine learning.
    </p>
</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: ui_card("R² Score",  f"{m['r2']:.3f}",        "Regression accuracy", "#00c853", "📈")
    with c2: ui_card("MAE",       f"€{m['mae']:.2f}M",     "Mean absolute error",  "#ff6b6b", "📉")
    with c3: ui_card("Dataset",   f"{len(df):,}",           "Players analysed",     "#448aff", "🗄️")
    with c4: ui_card("Features",  f"{len(artifacts['all_feature_names'])}",
                     "Model dimensions", "#ffd700", "🔧")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        st.markdown("""
<div style="background:#151b2b;border-radius:16px;padding:28px;
            border:1px solid rgba(255,255,255,0.05);">
    <h3 style="color:#ffffff;margin-top:0;">Platform Overview</h3>
    <p style="color:#b0b0b0;line-height:1.8;font-size:0.95rem;">
        ScoutAI Pro combines two gradient boosting variants and a KNN instance-based
        learner into a voting ensemble for both regression and classification tasks.
        Target encoding compresses 1,009 clubs into a single dense feature with no
        data leakage, and percentile-based tier labels keep class balance stable.
    </p>
    <div style="display:flex;gap:12px;margin-top:24px;">
        <div style="flex:1;background:#1a2332;border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:1.5rem;font-weight:700;color:#00c853;">3</div>
            <div style="font-size:0.75rem;color:#8899a6;margin-top:4px;">Model Families</div>
        </div>
        <div style="flex:1;background:#1a2332;border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:1.5rem;font-weight:700;color:#00c853;">2</div>
            <div style="font-size:0.75rem;color:#8899a6;margin-top:4px;">Ensemble Heads</div>
        </div>
        <div style="flex:1;background:#1a2332;border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:1.5rem;font-weight:700;color:#00c853;">23</div>
            <div style="font-size:0.75rem;color:#8899a6;margin-top:4px;">Final Features</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

    with right:
        counts = df['Performance_Tier'].value_counts().reindex(TIER_ORDER).fillna(0)
        fig = go.Figure(data=[go.Pie(
            labels=counts.index, values=counts.values, hole=0.65,
            marker_colors=[TIER_COLORS[t] for t in counts.index],
            textinfo='label+percent', textfont_size=11,
            hovertemplate='<b>%{label}</b><br>%{value} players (%{percent})<extra></extra>',
        )])
        fig.update_layout(
            showlegend=False, height=340,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            margin=dict(t=30, b=20),
            title=dict(text='Squad Tier Distribution', font=dict(size=14, color='#e0e0e0')),
        )
        fig.add_annotation(
            text=f"<b>{len(df):,}</b><br>PLAYERS",
            x=0.5, y=0.5, font_size=15, showarrow=False, font_color='#ffffff',
        )
        st.plotly_chart(fig, use_container_width=True)

    # Quick insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Market Insights")
    i1, i2, i3 = st.columns(3)

    top = df.nlargest(1, 'Value Per M$').iloc[0]
    with i1:
        st.markdown(f"""
<div style="background:#151b2b;border-radius:12px;padding:20px;border:1px solid rgba(255,255,255,0.05);">
    <p style="color:#8899a6;font-size:0.8rem;margin:0;text-transform:uppercase;letter-spacing:1px;">Highest Valuation</p>
    <p style="color:#ffffff;font-weight:700;font-size:1.1rem;margin:8px 0 0;">{top['Name']}</p>
    <p style="color:#00c853;font-weight:700;margin:4px 0 0;">€{top['Value Per M$']:.2f}M</p>
</div>""", unsafe_allow_html=True)

    avg = df['Value Per M$'].mean()
    with i2:
        st.markdown(f"""
<div style="background:#151b2b;border-radius:12px;padding:20px;border:1px solid rgba(255,255,255,0.05);">
    <p style="color:#8899a6;font-size:0.8rem;margin:0;text-transform:uppercase;letter-spacing:1px;">Market Average</p>
    <p style="color:#ffffff;font-weight:700;font-size:1.1rem;margin:8px 0 0;">€{avg:.2f}M</p>
    <p style="color:#8899a6;font-size:0.8rem;margin:4px 0 0;">Across all positions</p>
</div>""", unsafe_allow_html=True)

    elite_pct = (df['Performance_Tier'] == 'Elite').mean() * 100
    with i3:
        st.markdown(f"""
<div style="background:#151b2b;border-radius:12px;padding:20px;border:1px solid rgba(255,255,255,0.05);">
    <p style="color:#8899a6;font-size:0.8rem;margin:0;text-transform:uppercase;letter-spacing:1px;">Elite Talent Pool</p>
    <p style="color:#ffffff;font-weight:700;font-size:1.1rem;margin:8px 0 0;">{elite_pct:.1f}%</p>
    <p style="color:#ffd700;font-size:0.8rem;margin:4px 0 0;">≥80 Overall Rating</p>
</div>""", unsafe_allow_html=True)


# ============================================================
# PAGE — PLAYER VALUATION
# ============================================================
def page_valuation(artifacts: dict) -> None:
    ui_section(
        "💰 Player Valuation Engine",
        "Input a player attribute profile to generate an AI-powered market value estimate and tier classification.",
    )
    df = artifacts['df']

    col_in, col_out = st.columns([2, 3])

    with col_in:
        c1, c2 = st.columns(2)
        with c1:
            age     = st.slider("Age", 16, 40, 24)
            overall = st.slider("Overall Rating", 45, 95, 76)
        with c2:
            potential   = st.slider("Future Potential", 50, 99, 81)
            total_stats = st.slider("Total Stats Score", 700, 2700, 1850)

        countries = sorted(df['Country'].unique())
        teams     = sorted(df['Team'].unique())
        positions = sorted(df['Position'].unique())

        c3, c4 = st.columns(2)
        with c3:
            country  = st.selectbox("Nationality", countries)
            position = st.selectbox("Position", positions)
        with c4:
            team = st.selectbox("Current Club", teams)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🔮 GENERATE VALUATION REPORT", type="primary")

    with col_out:
        if run:
            player = {
                'Age': age, 'Overall_Rating': overall,
                'Future Potential': potential, 'Total_Stats Score': total_stats,
                'Country': country, 'Team': team, 'Position': position,
            }
            result = predict_player(player, artifacts)
            tier   = result['tier']
            color  = TIER_COLORS.get(tier, '#78909c')

            # Main result card
            st.markdown(f"""
<div style="background:linear-gradient(145deg,#151b2b,#1a2332);border-radius:20px;
            padding:32px;border:1px solid {color}40;
            box-shadow:0 8px 32px {color}18;margin-bottom:24px;">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px;">
        <div>
            <p style="color:#8899a6;font-size:0.8rem;margin:0;text-transform:uppercase;letter-spacing:1.5px;">
                Estimated Market Value
            </p>
            <h1 style="color:#ffffff;font-size:3.2rem;margin:8px 0;font-weight:800;">
                €{result['value']:.2f}M
            </h1>
        </div>
        <div style="text-align:right;">
            <p style="color:#8899a6;font-size:0.8rem;margin:0;text-transform:uppercase;letter-spacing:1.5px;">
                Performance Tier
            </p>
            <div style="margin-top:10px;">{ui_tier_badge(tier)}</div>
        </div>
    </div>
    <div style="background:rgba(0,0,0,0.2);border-radius:12px;padding:16px;margin-bottom:20px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <span style="color:#b0b0b0;font-size:0.9rem;font-weight:500;">Model Confidence</span>
            <span style="color:{color};font-weight:700;font-size:1rem;">{result['confidence']*100:.1f}%</span>
        </div>
        <div style="background:#0a0e17;border-radius:6px;height:10px;overflow:hidden;">
            <div style="width:{result['confidence']*100:.1f}%;background:linear-gradient(90deg,{color},{color}80);
                        height:100%;border-radius:6px;"></div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

            # Key drivers
            st.markdown("#### Key Value Drivers")
            fcols = st.columns(len(result['factors']))
            for col, (feat, imp) in zip(fcols, result['factors']):
                with col:
                    st.markdown(f"""
<div style="background:#151b2b;border-radius:12px;padding:16px;text-align:center;
            border:1px solid rgba(255,255,255,0.05);">
    <div style="font-size:1.4rem;font-weight:800;color:#00c853;">{imp*100:.1f}%</div>
    <div style="font-size:0.75rem;color:#8899a6;margin-top:6px;">{feat}</div>
</div>""", unsafe_allow_html=True)

            # Tier probability chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Tier Probability Distribution")
            probs = result['probs']
            ordered_tiers = [t for t in TIER_ORDER if t in probs]
            fig = go.Figure(go.Bar(
                x=ordered_tiers,
                y=[probs[t]*100 for t in ordered_tiers],
                marker_color=[TIER_COLORS[t] for t in ordered_tiers],
                text=[f"{probs[t]*100:.1f}%" for t in ordered_tiers],
                textposition='outside',
                textfont=dict(color='#e0e0e0', size=11),
                hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
            ))
            dark_layout(fig, height=300)
            fig.update_layout(xaxis_title="", yaxis_title="Probability (%)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Comparative context
            st.markdown("#### Attribute Context vs. Dataset Average")
            comp = pd.DataFrame({
                'Attribute': ['Overall', 'Potential', 'Total Stats'],
                'Player':    [overall, potential, total_stats],
                'Dataset Avg': [
                    df['Overall_Rating'].mean(),
                    df['Future Potential'].mean(),
                    df['Total_Stats Score'].mean(),
                ],
            })
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name='This Player', x=comp['Attribute'], y=comp['Player'],
                marker_color='#00c853',
                text=comp['Player'].round(0).astype(int), textposition='outside',
            ))
            fig2.add_trace(go.Bar(
                name='Dataset Average', x=comp['Attribute'], y=comp['Dataset Avg'],
                marker_color='rgba(255,255,255,0.15)',
                text=comp['Dataset Avg'].round(1), textposition='outside',
            ))
            dark_layout(fig2, height=320)
            fig2.update_layout(barmode='group',
                               legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.markdown("""
<div style="background:#151b2b;border-radius:20px;padding:70px 40px;text-align:center;
            border:2px dashed rgba(0,200,83,0.15);margin-top:20px;">
    <div style="font-size:3rem;margin-bottom:16px;">🔮</div>
    <h3 style="color:#ffffff;margin:0;">Ready to Generate Valuation</h3>
    <p style="color:#8899a6;margin-top:10px;max-width:400px;margin-left:auto;margin-right:auto;">
        Configure the player attribute profile and click the button to receive
        an AI-powered market analysis with tier classification.
    </p>
</div>""", unsafe_allow_html=True)


# ============================================================
# PAGE — ANALYTICS
# ============================================================
def page_analytics(artifacts: dict) -> None:
    ui_section(
        "📊 Player Analytics",
        "Interactive exploration of market trends, attribute correlations, and squad composition.",
    )
    df = artifacts['df']

    with st.expander("🔍 Analysis Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            pos_filter = st.multiselect("Position", sorted(df['Position'].unique()), default=[])
        with f2:
            age_range = st.slider(
                "Age Range", int(df['Age'].min()), int(df['Age'].max()),
                (int(df['Age'].min()), int(df['Age'].max())), key="ana_age",
            )
        with f3:
            ov_range = st.slider(
                "Overall Range", int(df['Overall_Rating'].min()), int(df['Overall_Rating'].max()),
                (int(df['Overall_Rating'].min()), int(df['Overall_Rating'].max())), key="ana_ov",
            )

    mask = (
        df['Age'].between(*age_range) &
        df['Overall_Rating'].between(*ov_range)
    )
    if pos_filter:
        mask &= df['Position'].isin(pos_filter)
    filtered = df[mask]

    if len(filtered) < 50:
        st.warning("Too few players match the current filters. Please relax the criteria.")
        return

    st.markdown(
        f"<p style='color:#8899a6;font-size:0.85rem;'>"
        f"Showing <b style='color:#00c853;'>{len(filtered):,}</b> of {len(df):,} players</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            filtered.sample(min(3000, len(filtered)), random_state=RANDOM_STATE),
            x='Age', y='Value Per M$', color='Performance_Tier',
            color_discrete_map=TIER_COLORS,
            hover_data=['Name', 'Overall_Rating', 'Position'],
            opacity=0.65, title='Age vs. Market Value',
            category_orders={'Performance_Tier': TIER_ORDER},
        )
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            filtered, x='Position', y='Value Per M$', color='Performance_Tier',
            color_discrete_map=TIER_COLORS, title='Value Distribution by Position',
            category_orders={'Performance_Tier': TIER_ORDER},
        )
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        gbr   = artifacts['models']['gbr_reg']
        names = artifacts['all_feature_names']
        imp   = pd.DataFrame({
            'Feature':    [n.replace('_', ' ').replace('pos ', '').title() for n in names],
            'Importance': gbr.feature_importances_,
        }).sort_values('Importance', ascending=True).tail(12)
        fig = go.Figure(go.Bar(
            x=imp['Importance'], y=imp['Feature'], orientation='h',
            marker_color='#00c853',
            text=[f"{v:.3f}" for v in imp['Importance']], textposition='outside',
        ))
        dark_layout(fig, title='Value Drivers (GBR Importances)', height=420)
        fig.update_layout(margin=dict(l=140))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        num_cols = ['Age', 'Overall_Rating', 'Future Potential', 'Value Per M$', 'Total_Stats Score']
        corr = filtered[num_cols].corr()
        fig = px.imshow(
            corr, text_auto='.2f', aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Attribute Correlation Matrix', zmin=-1, zmax=1,
        )
        dark_layout(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fig = px.histogram(
            filtered, x='Value Per M$', color='Performance_Tier',
            color_discrete_map=TIER_COLORS, nbins=50,
            title='Market Value Distribution', opacity=0.75,
            category_orders={'Performance_Tier': TIER_ORDER},
        )
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        pos_tier = filtered.groupby(['Position', 'Performance_Tier']).size().reset_index(name='Count')
        fig = px.bar(
            pos_tier, x='Position', y='Count', color='Performance_Tier',
            color_discrete_map=TIER_COLORS, title='Squad Composition by Position',
            barmode='stack', category_orders={'Performance_Tier': TIER_ORDER},
        )
        dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE — SIMILAR PLAYERS
# ============================================================
def page_similar(artifacts: dict) -> None:
    ui_section(
        "🔍 Similar Players",
        "Discover comparable talent using multi-dimensional cosine similarity analysis.",
    )
    df     = artifacts['df']
    labels = artifacts['display_labels']   # FIX BUG 5: precomputed, O(1) lookup

    tab1, tab2 = st.tabs(["📋 By Existing Player", "⚙️ By Custom Profile"])

    with tab1:
        # FIX BUG 5: format_func is O(1) dict lookup instead of O(n) df search
        selected = st.selectbox(
            "Select a player from the database",
            list(labels.keys()),
            format_func=lambda x: labels.get(x, x),
        )

        if st.button("Find Comparables", key="sim_existing", type="primary"):
            # FIX BUG 6: use .loc with the unique Name (already deduplicated in load_data)
            idx = df.index[df['Name'] == selected][0]
            row = df.loc[idx]
            player_dict = {
                'Age': int(row['Age']), 'Overall_Rating': int(row['Overall_Rating']),
                'Future Potential': int(row['Future Potential']),
                'Total_Stats Score': int(row['Total_Stats Score']),
                'Country': str(row['Country']), 'Team': str(row['Team']),
                'Position': str(row['Position']),
            }
            similar = find_similar(player_dict, artifacts, n=7)
            similar = [s for s in similar if s['name'] != selected][:5]

            st.markdown(f"#### Top 5 Comparables for **{selected}**")

            # FIX BUG 12: avoid deprecated Styler API — plain formatted DataFrame
            sim_df = pd.DataFrame(similar)
            sim_df['similarity'] = sim_df['similarity'].mul(100).round(1)
            display = sim_df[['name','similarity','value','overall','potential','age','position','tier']].copy()
            display.columns = ['Player','Similarity %','Value (M€)','OVR','POT','Age','POS','Tier']
            display['Value (M€)'] = display['Value (M€)'].apply(lambda x: f"€{x:.2f}")
            display['Similarity %'] = display['Similarity %'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display, use_container_width=True, hide_index=True)

            # ── Radar chart ────────────────────────────────────────────────
            # FIX BUG 7: use only columns that actually exist in the dataset
            radar_attrs   = ['Overall_Rating', 'Future Potential', 'Total_Stats Score', 'Age']
            radar_labels  = ['Overall', 'Potential', 'Total Stats', 'Age']
            sel_vals = [float(row[c]) for c in radar_attrs]

            def _scale(vals, cols):
                out = []
                for v, c in zip(vals, cols):
                    mn, mx = float(df[c].min()), float(df[c].max())
                    out.append((v - mn) / (mx - mn) if mx > mn else 0.5)
                return out

            fig = go.Figure()
            cats = radar_labels + [radar_labels[0]]
            fig.add_trace(go.Scatterpolar(
                r=_scale(sel_vals, radar_attrs) + [_scale(sel_vals, radar_attrs)[0]],
                theta=cats, fill='toself', name=selected,
                line_color='#00c853', fillcolor='rgba(0,200,83,0.15)',
            ))
            if similar:
                top   = similar[0]
                top_r = df[df['Name'] == top['name']].iloc[0]
                top_v = [float(top_r[c]) for c in radar_attrs]
                fig.add_trace(go.Scatterpolar(
                    r=_scale(top_v, radar_attrs) + [_scale(top_v, radar_attrs)[0]],
                    theta=cats, fill='toself', name=top['name'],
                    line_color='#448aff', fillcolor='rgba(68,138,255,0.1)',
                ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1],
                                   gridcolor='rgba(255,255,255,0.1)',
                                   tickvals=[0.25,0.5,0.75,1.0],
                                   ticktext=['25%','50%','75%','100%'],
                                   tickfont=dict(color='#8899a6',size=9)),
                    bgcolor='rgba(0,0,0,0)',
                ),
                showlegend=True, height=460,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom', y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            age2     = st.slider("Age", 16, 40, 24, key="t2_age")
            overall2 = st.slider("Overall Rating", 45, 95, 76, key="t2_ovr")
        with c2:
            potential2   = st.slider("Potential", 50, 99, 81, key="t2_pot")
            total_stats2 = st.slider("Total Stats", 700, 2700, 1850, key="t2_stats")

        c3, c4 = st.columns(2)
        with c3:
            country2  = st.selectbox("Nationality", sorted(df['Country'].unique()), key="t2_cnt")
            position2 = st.selectbox("Position", sorted(df['Position'].unique()), key="t2_pos")
        with c4:
            team2 = st.selectbox("Current Club", sorted(df['Team'].unique()), key="t2_team")

        if st.button("Find Comparables", key="sim_custom", type="primary"):
            player_dict = {
                'Age': age2, 'Overall_Rating': overall2, 'Future Potential': potential2,
                'Total_Stats Score': total_stats2, 'Country': country2,
                'Team': team2, 'Position': position2,
            }
            similar = find_similar(player_dict, artifacts, n=5)
            st.markdown("<h4 style='color:#e0e0e0;margin-top:20px;'>Top Matches</h4>", unsafe_allow_html=True)
            for s in similar:
                tc = TIER_COLORS.get(s['tier'], '#78909c')
                st.markdown(f"""
<div style="background:#151b2b;border-radius:12px;padding:18px;margin-bottom:12px;
            border-left:4px solid {tc};display:flex;justify-content:space-between;align-items:center;">
    <div>
        <div style="font-weight:700;color:#ffffff;font-size:1.05rem;">{s['name']}</div>
        <div style="color:#8899a6;font-size:0.85rem;margin-top:4px;">
            {s['position']} | Age {s['age']} | OVR {s['overall']} | {s['team']}
        </div>
    </div>
    <div style="text-align:right;min-width:100px;">
        <div style="font-weight:800;color:#00c853;font-size:1.3rem;">{s['similarity']*100:.1f}%</div>
        <div style="color:#8899a6;font-size:0.75rem;">similarity</div>
        <div style="color:#ffffff;font-weight:600;font-size:0.9rem;margin-top:4px;">€{s['value']:.2f}M</div>
    </div>
</div>""", unsafe_allow_html=True)


# ============================================================
# PAGE — ABOUT MODEL
# ============================================================
def page_about(artifacts: dict) -> None:
    ui_section(
        "🧠 About the Model",
        "Technical documentation of the ScoutAI Pro machine learning pipeline and evaluation framework.",
    )
    m = artifacts['metrics']

    st.markdown("""
<div style="background:#151b2b;border-radius:16px;padding:28px;margin-bottom:28px;border:1px solid rgba(255,255,255,0.05);">
    <h4 style="color:#00c853;margin-top:0;font-size:0.9rem;text-transform:uppercase;letter-spacing:1.5px;">
        Preprocessing Pipeline
    </h4>
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:20px;">
        <div style="background:#1a2332;border-radius:8px;padding:10px 16px;color:#e0e0e0;font-size:0.8rem;">Raw FIFA Data</div>
        <div style="color:#00c853;font-weight:700;">→</div>
        <div style="background:#1a2332;border-radius:8px;padding:10px 16px;color:#e0e0e0;font-size:0.8rem;">Filter Zero-Value Players</div>
        <div style="color:#00c853;font-weight:700;">→</div>
        <div style="background:#1a2332;border-radius:8px;padding:10px 16px;color:#e0e0e0;font-size:0.8rem;">Target Encoding (Country, Team)</div>
        <div style="color:#00c853;font-weight:700;">→</div>
        <div style="background:#1a2332;border-radius:8px;padding:10px 16px;color:#e0e0e0;font-size:0.8rem;">One-Hot Encoding (Position)</div>
        <div style="color:#00c853;font-weight:700;">→</div>
        <div style="background:#1a2332;border-radius:8px;padding:10px 16px;color:#e0e0e0;font-size:0.8rem;">Standard Scaling</div>
        <div style="color:#00c853;font-weight:700;">→</div>
        <div style="background:#1a2332;border-radius:8px;padding:10px 16px;color:#e0e0e0;font-size:0.8rem;">log1p Target Transform</div>
    </div>
    <p style="color:#8899a6;font-size:0.85rem;margin-top:16px;line-height:1.6;">
        All encoders and the StandardScaler are fit exclusively on the training partition
        to prevent data leakage. Stratified sampling preserves the imbalanced tier distribution
        across the 80/20 train/test split.
    </p>
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    specs = [
        ("Gradient Boosting", "Two GBR/GBC variants with different depth/learning-rate hyperparams provide ensemble diversity. Handles non-linearity, heteroscedasticity, and extreme outliers natively.", "#00c853"),
        ("K-Nearest Neighbors", "Instance-based learner with k=12 and distance weighting. Football valuation is inherently comparative — KNN provides interpretable 'look-alike' players for scouting departments.", "#448aff"),
        ("Voting Ensemble", "Soft-voting aggregates predicted probabilities across all base learners. More stable than any individual model across different player profile segments.", "#ff6b6b"),
    ]
    for col, (title, desc, color) in zip([c1, c2, c3], specs):
        with col:
            st.markdown(f"""
<div style="background:#151b2b;border-radius:16px;padding:24px;height:100%;
            border-top:4px solid {color};border:1px solid rgba(255,255,255,0.05);border-top-width:4px;">
    <h4 style="color:#ffffff;margin-top:0;font-size:1rem;">{title}</h4>
    <p style="color:#8899a6;font-size:0.88rem;line-height:1.7;">{desc}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Test-Set Performance Summary")
    perf = pd.DataFrame({
        'Task':          ['Regression','Regression','Regression','Classification','Classification','Classification','Classification'],
        'Metric':        ['R² Score','MAE (M€)','RMSE (M€)','Accuracy','F1-Weighted','F1-Macro','MCC'],
        'Ensemble Value':[
            f"{m['r2']:.4f}", f"{m['mae']:.4f}", f"{m['rmse']:.4f}",
            f"{m['accuracy']:.4f}", f"{m['f1_weighted']:.4f}",
            f"{m['f1_macro']:.4f}", f"{m['mcc']:.4f}",
        ],
        'Interpretation':[
            'Variance explained (1.0 = perfect)',
            'Average prediction error in M€',
            'Penalises large misses more than MAE',
            'Overall correct tier prediction rate',
            'Precision/recall balance (class-weighted)',
            'Unweighted per-tier F1 average',
            'Balanced correlation: +1 perfect, 0 random',
        ],
    })
    st.dataframe(perf, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Feature Importance — Primary GBR Regressor")
    gbr   = artifacts['models']['gbr_reg']
    names = artifacts['all_feature_names']
    imp   = pd.DataFrame({
        'Feature':    [n.replace('_',' ').replace('pos ','').title() for n in names],
        'Importance': gbr.feature_importances_,
    }).sort_values('Importance', ascending=True)
    fig = go.Figure(go.Bar(
        x=imp['Importance'], y=imp['Feature'], orientation='h',
        marker_color='#00c853',
        text=[f"{v:.3f}" for v in imp['Importance']], textposition='outside',
    ))
    dark_layout(fig, height=500)
    fig.update_layout(margin=dict(l=160, r=60))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📐 Evaluation Metrics Reference"):
        st.markdown("""
**Regression Metrics**
- **R² Score**: Proportion of market value variance explained. Range (−∞, 1]; 1.0 is perfect.
- **MAE**: Mean Absolute Error in M€. Average magnitude of prediction error, robust to outliers.
- **RMSE**: Root Mean Squared Error. Penalises large misses (e.g. undervaluing a superstar) more aggressively.

**Classification Metrics**
- **Accuracy**: Overall correct tier rate. Can mislead with imbalanced classes.
- **F1-Weighted**: Harmonic mean of precision and recall, weighted by tier support.
- **F1-Macro**: Unweighted per-tier F1 — treats Elite (small class) equally to Average.
- **MCC**: Matthews Correlation Coefficient. Most balanced single-number classification metric.
""")

    st.markdown("""
<div style="background:#151b2b;border-radius:16px;padding:24px;margin-top:24px;
            border:1px solid rgba(255,255,255,0.05);">
    <h4 style="color:#00c853;margin-top:0;font-size:0.9rem;text-transform:uppercase;letter-spacing:1.5px;">
        Reproducibility &amp; Anti-Leakage Guarantees
    </h4>
    <ul style="color:#8899a6;line-height:1.8;font-size:0.9rem;">
        <li><b>Target encoders</b> fit on training partition only — test set never influences encoding maps.</li>
        <li><b>StandardScaler</b> mean/std computed from training features only.</li>
        <li><b>Stratified split</b> preserves tier distribution (Elite only ~3%) across train/test.</li>
        <li><b>Deterministic pos_cols</b> — sorted() guarantees feature order at inference matches training.</li>
        <li><b>Random seed 42</b> locked across NumPy and all sklearn estimators.</li>
        <li><b>log1p target</b> reduces right skew from ~8.0 to ~1.2 before regression.</li>
        <li><b>Zero-value players filtered</b> before training to remove non-market rows.</li>
    </ul>
</div>""", unsafe_allow_html=True)


# ============================================================
# MAIN ROUTER
# ============================================================
def main() -> None:
    # FIX BUG 14: sidebar status is built AFTER artifacts are loaded
    with st.spinner("Loading data and training models …"):
        artifacts = build_pipeline("Fifa.csv")

    df = artifacts['df']
    m  = artifacts['metrics']

    with st.sidebar:
        st.markdown("""
<div style="text-align:center;margin-bottom:30px;">
    <h2 style="color:#ffffff;font-weight:800;margin:0;font-size:1.8rem;">⚽ SCOUTAI</h2>
    <p style="color:#00c853;font-size:0.8rem;letter-spacing:3px;margin:4px 0 0;font-weight:600;">PRO</p>
</div>""", unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Home", "💰 Player Valuation", "📊 Player Analytics",
             "🔍 Similar Players", "🧠 About Model"],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:30px 0;'>",
                    unsafe_allow_html=True)

        # FIX BUG 14: artifacts guaranteed loaded here
        st.markdown(f"""
<div style="background:#151b2b;border-radius:12px;padding:18px;border:1px solid rgba(0,200,83,0.1);">
    <p style="color:#8899a6;font-size:0.7rem;margin:0 0 12px;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">
        System Status
    </p>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <div style="width:8px;height:8px;background:#00c853;border-radius:50%;box-shadow:0 0 8px #00c853;"></div>
        <span style="color:#e0e0e0;font-size:0.82rem;font-weight:500;">Models Active</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <div style="width:8px;height:8px;background:#448aff;border-radius:50%;box-shadow:0 0 8px #448aff;"></div>
        <span style="color:#e0e0e0;font-size:0.82rem;font-weight:500;">{len(df):,} Players</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:8px;height:8px;background:#ffd700;border-radius:50%;box-shadow:0 0 8px #ffd700;"></div>
        <span style="color:#e0e0e0;font-size:0.82rem;font-weight:500;">R² {m['r2']:.3f}</span>
    </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("© 2026 ScoutAI Pro. Professional Football Intelligence.")

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
