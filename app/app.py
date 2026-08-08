import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Fraud Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# GLOBAL STYLING
# =============================================================================
st.markdown('''
<style>

/* ================================
   FRAUDSHIELD AI - FINTECH THEME
================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Main background */
.stApp {
    background:
    radial-gradient(circle at top left, rgba(220,38,38,0.20), transparent 35%),
    radial-gradient(circle at bottom right, rgba(127,29,29,0.20), transparent 30%),
    #0a0000;
    color: #f8fafc;
}


/* ================================
   SIDEBAR
================================ */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #150000,
        #050000
    );
    border-right: 1px solid rgba(220,38,38,0.25);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}


/* ================================
   HERO HEADER
================================ */

.hero {

    margin-top:-2rem;

    background:
    linear-gradient(
        135deg,
        rgba(153,27,27,0.6),
        rgba(0,0,0,0.6)
    );

    backdrop-filter: blur(20px);

    border:
    1px solid rgba(239,68,68,0.4);

    border-radius:24px;

    padding:35px;

    margin-bottom:30px;

    box-shadow:
    0 20px 60px rgba(220,38,38,0.3);

}


.hero h1 {

    font-size:42px;

    font-weight:800;

    color:white;
}



.hero p {

    color:#cbd5e1;

    font-size:17px;

}


/* ================================
   SECTION TITLES
================================ */

.section-title {

    font-size:28px;

    font-weight:800;

    background:
    linear-gradient(
        90deg,
        #ff1a1a,
        #7f1d1d
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

    margin-top:25px;

    margin-bottom:20px;

}


/* ================================
   GLASS CARDS
================================ */

.glass-card {

    background:
    linear-gradient(
        145deg,
        rgba(45,0,0,0.85),
        rgba(10,0,0,0.95)
    );

    border:
    1px solid rgba(239,68,68,0.22);

    border-radius:22px;

    padding:25px;

    box-shadow:
    0 20px 40px rgba(0,0,0,0.35);

    color:white !important;

}


.glass-card h3,
.glass-card p,
.glass-card b {

    color:white !important;

}


/* ================================
   INPUTS
================================ */


div[data-baseweb="input"] {

    background:#160404;

    border-radius:12px;

}


div[data-baseweb="select"] > div {

    background:#160404;

    border-radius:12px;

    color:white;

}


/* ================================
   BUTTONS
================================ */

.stButton button {

    width:100%;

    border-radius:14px;

    border:none;

    padding:12px 20px;

    font-weight:700;

    color:white;

    background:

    linear-gradient(
        135deg,
        #7f0000,
        #dc2626
    );

    transition:0.25s;

}


.stButton button:hover {

    transform:translateY(-3px);

    box-shadow:

    0 10px 25px rgba(220,38,38,0.5);

}


/* ================================
   METRICS
================================ */

div[data-testid="stMetric"] {

    background:

    rgba(20,0,0,0.8);

    border:

    1px solid rgba(239,68,68,0.18);

    border-radius:18px;

    padding:20px;

}


div[data-testid="stMetric"] label {

    color:#94a3b8 !important;

}


div[data-testid="stMetric"] [data-testid="stMetricValue"] {

    color:white;

}


/* ================================
   TABS
================================ */

.stTabs [data-baseweb="tab-list"] {

    gap:12px;

}


.stTabs [data-baseweb="tab"] {

    background:#140404;

    border-radius:12px;

    padding:12px 20px;

    color:#b3a3a3;

}


.stTabs [aria-selected="true"] {

    background:
    linear-gradient(
        135deg,
        #7f0000,
        #ff1a1a
    ) !important;

    color:white !important;

}
/* ================================
   HEADER BADGES
================================ */

.badges {
    margin-top: 20px;
}


.pill {

    display:inline-block;

    padding:8px 16px;

    margin:5px;

    border-radius:30px;

    background:
    rgba(0,0,0,0.35);

    border:
    1px solid rgba(239,68,68,0.4);

    color:white;

    font-size:14px;

    font-weight:600;

    backdrop-filter:blur(10px);

}


/*================================
   FEATURE LIST ITEMS
================================ */

.feature-chip {
    display:flex;
    align-items:center;
    gap:10px;

    padding:14px 18px;
    margin-bottom:12px;

    border-radius:14px;

    background:rgba(25,0,0,0.95) !important;

    border:1px solid rgba(239,68,68,0.2);

    border-left:4px solid #ff1a1a;

    color:#ffffff !important;

    font-size:15px;
    font-weight:600;
}


.feature-chip,
.feature-chip *,
.feature-chip p,
.feature-chip span,
.feature-chip div {

    color:#ffffff !important;

}




/* ================================
   REASON TAGS
================================ */

.reason-pill {

    display:inline-block;

    padding:7px 14px;

    margin:5px;

    border-radius:20px;

    background:

    rgba(255,255,255,0.15);

    border:

    1px solid rgba(255,255,255,0.25);

    color:white;

    font-size:13px;

    font-weight:600;

}


/* ================================
   RESULT BOXES
================================ */

.fraud-box {

    background:

    linear-gradient(
        135deg,
        #3d0000,
        #dc2626
    );

    border: 1px solid rgba(255,26,26,0.5);

    box-shadow: 0 15px 40px rgba(220,38,38,0.35);

    border-radius:24px;

    padding:30px;

    color:white;

}


.legit-box {

    background:

    linear-gradient(
        135deg,
        #0f1f14,
        #16a34a
    );

    border: 1px solid rgba(34,197,94,0.4);

    border-radius:24px;

    padding:30px;

    color:white;

}


/* Warning */

.warning-box {

    background:#1a0505;

    border-left:

    5px solid #ff4500;

    padding:18px;

    border-radius:14px;

    color:#f8fafc;

}
/* Streamlit markdown text fix */

.stMarkdown p,
.stMarkdown li {

    color:#e5e7eb;

}





.hero h1,
.glass-card h3,
.section-title {
    color:white !important;
}


/* Remove weird Streamlit spacing */

/* Remove the empty top gap */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem;
}

/* Remove extra space above the first element */
div[data-testid="stAppViewContainer"] .main {
    padding-top: 0rem !important;
}

/* Hide Streamlit's top header area */
header[data-testid="stHeader"] {
    height: 0rem !important;
    background: transparent !important;
}

div[data-testid="stToolbar"] {
    display: none !important;
}
div[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
}

.main .block-container {
    padding-top: 0 !important;
}
div[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(220,38,38,0.20), transparent 35%),
        radial-gradient(circle at bottom right, rgba(127,29,29,0.20), transparent 30%),
        #0a0000 !important;
}
/* REMOVE STREAMLIT TOP HEADER GAP COMPLETELY */



[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
}

[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}

section.main > div.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
/* Force sidebar to be visible */
section[data-testid="stSidebar"] {
    display: block !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
    width: 21rem !important;
    min-width: 21rem !important;
}

section[data-testid="stSidebarCollapsedControl"] {
    display: block !important;
}
</style>
''', unsafe_allow_html=True)

# =============================================================================
# HERO HEADER
# =============================================================================
# =============================================================================
# DASHBOARD HEADER
# =============================================================================

st.markdown("""
<div class="hero">

<h1>
🔍 FraudShield AI
</h1>

<p>
Real-Time Financial Fraud Detection & Risk Intelligence Platform
</p>

<div class="badges">

<span class="pill">
⚡ XGBoost Engine
</span>

<span class="pill">
📊 6.3M Transactions
</span>

<span class="pill">
🎯 99.9% ROC-AUC
</span>

<span class="pill">
🕵️ Explainable AI
</span>

</div>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODEL LOADING
# The notebook saves the model with: joblib.dump(model, "xgboost.pkl")
# which lands in the notebook's working directory, NOT "../models/".
# We try a few sensible locations so the app works wherever it's run from.
# ---------------------------------------------------------------------------
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _script_dir = os.getcwd()

CANDIDATE_PATHS = [
    "xgboost.pkl",
    os.path.join(_script_dir, "xgboost.pkl"),
    "../xgboost.pkl",
    "models/xgboost.pkl",
    "../models/xgboost.pkl",
]

model = None
model_loaded = False
model_path_used = None

for path in CANDIDATE_PATHS:
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            model_loaded = True
            model_path_used = path
            break
        except Exception as e:
            st.sidebar.error(f"Found {path} but failed to load it: {e}")

# =============================================================================
# SIDEBAR DASHBOARD
# =============================================================================

st.sidebar.markdown("""
<div style="
font-size:28px;
font-weight:800;
margin-bottom:15px;
">
🛡️ FraudShield AI
</div>
""", unsafe_allow_html=True)


st.sidebar.markdown("## ⚙️ System Status")

if not model_loaded:

    st.sidebar.error(
        "🚫 Model unavailable\n\n"
        "Checked paths:\n"
        + "\n".join(CANDIDATE_PATHS)
    )

else:

    st.sidebar.success(
        "🟢 Model Online"
    )

    st.sidebar.caption(
        f"Loaded: {model_path_used}"
    )


st.sidebar.markdown("---")


st.sidebar.markdown("## 🧠 Model Information")

st.sidebar.markdown("""
<div class="glass-card">

<b>Algorithm</b><br>
🌲 XGBoost<br><br>

<b>Dataset</b><br>
📦 PaySim<br><br>

<b>Transaction Types</b><br>
🔁 TRANSFER<br>
💸 CASH_OUT

</div>
""", unsafe_allow_html=True)


THRESHOLD = 0.4


st.sidebar.markdown("---")

st.sidebar.markdown("## 🎚️ Risk Threshold")

st.sidebar.progress(THRESHOLD)


st.sidebar.caption(
    f"Transactions above {THRESHOLD:.0%} risk are flagged."
)


st.sidebar.caption(
    "Balance anomaly rules are also applied."
)

# The model was trained ONLY on TRANSFER and CASH_OUT transactions
VALID_TYPES = ["TRANSFER", "CASH_OUT"]

FEATURE_COLUMNS = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'errorBalanceOrig',
    'errorBalanceDest', 'hourOfDay', 'origEmptied', 'type_TRANSFER'
]

tab1, tab2, tab3 = st.tabs(["📖 About", "⚡ Live Prediction", "📊 Results"])

# ==================== TAB 1: ABOUT ====================

with tab1:

    st.markdown(
        '<div class="section-title">🛡️ Intelligent Fraud Detection System</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown("""
        <div class="glass-card">

        <h3>🎯 The Problem</h3>

        <p>
        Digital payment fraud creates billions in financial losses.
        Traditional accuracy metrics fail because fraud cases are rare.
        </p>

        </div>
        """, unsafe_allow_html=True)



    with col2:

        st.markdown("""
        <div class="glass-card">

        <h3>🤖 Our Solution</h3>

        <p>
        An XGBoost-powered AI system that evaluates transactions
        in real-time and explains suspicious behaviour.
        </p>

        </div>
        """, unsafe_allow_html=True)



    with col3:

        st.markdown("""
        <div class="glass-card">

        <h3>⚡ Real-Time Scoring</h3>

        <p>
        Every transaction receives a fraud risk score with
        actionable recommendations.
        </p>

        </div>
        """, unsafe_allow_html=True)



    st.markdown(
        '<div class="section-title">🧠 AI Pipeline</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("""
        <div class="glass-card">

        <h3>📊 Dataset & Training</h3>

        <div class="feature-chip">
        📦 PaySim dataset - 6.3M transactions
        </div>

        <div class="feature-chip">
        🌲 XGBoost classification model
        </div>

        <div class="feature-chip">
        ⚖️ Class imbalance handling
        </div>

        <div class="feature-chip">
        🔍 Explainable prediction logic
        </div>

        </div>
        """, unsafe_allow_html=True)



    with col2:

        st.markdown("""
        <div class="glass-card">

        <h3>🚨 Fraud Patterns Detected</h3>

        <div class="feature-chip">
        🕳️ Account emptied instantly
        </div>

        <div class="feature-chip">
        ⚠️ Balance mismatch detected
        </div>

        <div class="feature-chip">
        💸 Suspicious transfer behaviour
        </div>

        <div class="feature-chip">
        💰 High-value transactions
        </div>

        </div>
        """, unsafe_allow_html=True)



    st.markdown(
        '<div class="section-title">🏆 Model Highlights</div>',
        unsafe_allow_html=True
    )


    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:
        st.metric(
            "ROC-AUC",
            "99.9%"
        )


    with metric2:
        st.metric(
            "Transactions",
            "6.3M+"
        )


    with metric3:
        st.metric(
            "Recall",
            "99.8%"
        )


    with metric4:
        st.metric(
            "Model",
            "XGBoost"
        )

# ==================== TAB 2: LIVE PREDICTION ====================
with tab2:
    st.markdown('<div class="section-title">Try a Transaction</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    <b>⚡ Quick Test Cases</b><br>
    🚨 <b>FRAUD:</b> TRANSFER 1000, Sender 5000→0, Receiver 0→1000 (account emptied)<br>
    ✅ <b>LEGIT:</b> TRANSFER 1000, Sender 5000→4000, Receiver 0→1000 (normal)
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    if not model_loaded:
        st.warning("Model not loaded. Please check the model file path in the sidebar message above.")
    else:
        defaults = {
            "amount": 1000.0, "old_orig": 5000.0, "new_orig": 4000.0,
            "old_dest": 0.0, "new_dest": 1000.0, "txn_type": "TRANSFER",
            "hour": 12,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚨 Test Fraud", use_container_width=True):
                st.session_state.amount = 1000.0
                st.session_state.old_orig = 5000.0
                st.session_state.new_orig = 0.0
                st.session_state.old_dest = 0.0
                st.session_state.new_dest = 1000.0
                st.session_state.txn_type = "TRANSFER"
                st.rerun()
        with col2:
            if st.button("✅ Test Legit", use_container_width=True):
                st.session_state.amount = 1000.0
                st.session_state.old_orig = 5000.0
                st.session_state.new_orig = 4000.0
                st.session_state.old_dest = 0.0
                st.session_state.new_dest = 1000.0
                st.session_state.txn_type = "TRANSFER"
                st.rerun()
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.rerun()

        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("💵 Transaction Amount", min_value=0.0, step=100.0, key="amount")
            old_orig = st.number_input("👤 Sender Balance BEFORE", min_value=0.0, step=100.0, key="old_orig")
            new_orig = st.number_input("👤 Sender Balance AFTER", min_value=0.0, step=100.0, key="new_orig")
        with col2:
            old_dest = st.number_input("🏦 Receiver Balance BEFORE", min_value=0.0, step=100.0, key="old_dest")
            new_dest = st.number_input("🏦 Receiver Balance AFTER", min_value=0.0, step=100.0, key="new_dest")
            txn_type = st.selectbox("🔀 Transaction Type", VALID_TYPES, key="txn_type")

        hour = st.slider("🕐 Hour of Day (used as 'step' proxy)", 0, 23, key="hour")

        if st.button("🔎 Check Transaction", type="primary", use_container_width=True):
            errorBalanceOrig = new_orig + amount - old_orig
            errorBalanceDest = old_dest + amount - new_dest
            origEmptied = 1 if new_orig == 0 else 0
            hourOfDay = hour
            step = hour  # step % 24 == hourOfDay in training; using hour directly is a reasonable proxy
            type_TRANSFER = 1 if txn_type == "TRANSFER" else 0

            with st.expander("🧾 Debug Info"):
                st.write(f"- Balance Error (Sender): {errorBalanceOrig:.2f}")
                st.write(f"- Balance Error (Receiver): {errorBalanceDest:.2f}")
                st.write(f"- Sender Emptied: {origEmptied}")
                st.write(f"- Transaction Type: {txn_type} (type_TRANSFER={type_TRANSFER})")

            row = pd.DataFrame([[
                step, amount, old_orig, new_orig, old_dest, new_dest,
                errorBalanceOrig, errorBalanceDest, hourOfDay, origEmptied, type_TRANSFER
            ]], columns=FEATURE_COLUMNS)

            try:
                with st.spinner("Scoring transaction..."):
                    time.sleep(0.4)
                    prob = model.predict_proba(row)[0][1]

                st.markdown("---")
                st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

                reasons = []
                if origEmptied == 1:
                    reasons.append("Sender's account was emptied to zero")
                if abs(errorBalanceOrig) != 0:
                    reasons.append("Balance inconsistency detected (amount ≠ balance change)")
                if txn_type in ["TRANSFER", "CASH_OUT"]:
                    reasons.append("High-risk transaction type")
                if amount > 10000:
                    reasons.append("Unusually large transaction")

                # Same decision logic as the notebook:
                # 1) model probability >= THRESHOLD -> fraud
                # 2) otherwise, a negative sender balance error still overrides to fraud
                # 3) otherwise -> legit
                is_fraud = prob >= THRESHOLD
                override_flag = False
                if not is_fraud and errorBalanceOrig < 0:
                    is_fraud = True
                    override_flag = True
                    reasons.append("Negative balance error override (money vanished from ledger)")

                result_col, gauge_col = st.columns([1.2, 1])

                with result_col:
                    if is_fraud:
                        pills = "".join([f'<span class="reason-pill">🔸 {r}</span>' for r in reasons]) \
                            or '<span class="reason-pill">Model risk score exceeded threshold</span>'
                        override_note = (
                            '<p style="font-size:13px; opacity:0.85;">⚑ Flagged by balance-error override rule, '
                            'not the probability threshold.</p>' if override_flag else ""
                        )
                        st.markdown(f"""
                        <div class="fraud-box">
                        <h2>⚠️ LIKELY FRAUD</h2>
                        <p style="font-size: 26px; font-weight: 800;">Risk Score: {prob:.2%}</p>
                        <p style="font-weight: 700;">🛑 Recommended action: BLOCK &amp; flag for review</p>
                        {override_note}
                        <hr style="border-color: rgba(255,255,255,0.3);">
                        <p><b>Why flagged:</b></p>
                        {pills}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="legit-box">
                        <h2>✅ LOOKS LEGITIMATE</h2>
                        <p style="font-size: 26px; font-weight: 800;">Risk Score: {prob:.2%}</p>
                        <p style="font-weight: 700;">🟢 Transaction appears safe</p>
                        </div>
                        """, unsafe_allow_html=True)

                with gauge_col:
                    gauge_color = "#ff1a1a" if is_fraud else "#16a34a"
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={'suffix': "%", 'font': {'size': 34, 'color': '#f8fafc'}},
                        title={'text': "Fraud Risk Score", 'font': {'size': 16, 'color': '#f8fafc'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#f8fafc'},
                            'bar': {'color': gauge_color},
                            'bgcolor': '#0a0000',
                            'bordercolor': 'rgba(239,68,68,0.3)',
                            'steps': [
                                {'range': [0, THRESHOLD * 100], 'color': 'rgba(22,163,74,0.18)'},
                                {'range': [THRESHOLD * 100, 100], 'color': 'rgba(220,38,38,0.25)'},
                            ],
                            'threshold': {
                                'line': {'color': "#ff1a1a", 'width': 3},
                                'thickness': 0.8,
                                'value': THRESHOLD * 100
                            }
                        }
                    ))
                    fig.update_layout(
                        height=260,
                        margin=dict(l=20, r=20, t=40, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#f8fafc'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    if override_flag:
                        st.caption("⚑ Gauge shows raw model probability — the final verdict above was flipped by the override rule.")

                st.markdown('<div class="section-title">Key Indicators</div>', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Balance Error (Sender)", f"{errorBalanceOrig:.2f}")
                with col2:
                    st.metric("Balance Error (Receiver)", f"{errorBalanceDest:.2f}")
                with col3:
                    st.metric("Sender Emptied", "Yes 🕳️" if origEmptied == 1 else "No")
                with col4:
                    st.metric("Transaction Type", txn_type)

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ==================== TAB 3: RESULTS ====================
with tab3:
    st.markdown('<div class="section-title">Model Results & Explainability</div>', unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Precision": [0.778, 1.0, 0.991],
        "Recall": [0.640, 0.740, 0.998],
        "F1": [0.703, 0.851, 0.994],
        "ROC-AUC": [0.996, 0.992, 0.999],
        "PR-AUC": [0.78, 0.85, 0.99],
    })

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("#### 🏆 Model Comparison")
        st.dataframe(
            metrics_df.style.background_gradient(cmap="Reds", subset=["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
                             .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_bar = px.bar(
            melted, x="Metric", y="Score", color="Model", barmode="group",
            color_discrete_sequence=["#ff1a1a", "#7f1d1d", "#4b5563"],
            title="Metric Comparison Across Models"
        )
        fig_bar.update_layout(
            height=380,
            legend_title_text="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#f8fafc'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Key Metrics")
        st.metric("Fraud Detection Rate", "99.76%", delta="High recall")
        st.metric("False Alarm Rate", "0.003%", delta="-Low", delta_color="inverse")
        st.metric("Overall Accuracy", "99.8%", delta="But misleading for fraud!", delta_color="off")

        st.markdown("#### 🥧 Precision vs Recall (XGBoost)")
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Precision", "Recall", "Miss"],
            values=[0.991, 0.998, 1 - 0.998],
            hole=0.55,
            marker_colors=["#ff1a1a", "#7f1d1d", "#2a2a2a"]
        )])
        fig_pie.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#f8fafc'},
            legend={'font': {'color': '#f8fafc'}}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
