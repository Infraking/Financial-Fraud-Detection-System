"""
FRAUDX — Financial Risk Intelligence & Risk Monitoring Platform
PISB TechRush 2026 Submission

Detect. Explain. Prioritize. Investigate.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as bg
import streamlit as st

# Configure page layout and title
st.set_page_config(
    page_title="FRAUDX — Financial Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure src modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_loader import load_xgboost_model, get_model_schema
from src.preprocessing import prepare_transaction_features
from src.prediction import analyze_transaction
from src.shap_explainer import explain_transaction, get_global_shap_importance, FEATURE_DISPLAY_NAMES
from src.risk_scoring import compute_risk_radar_dimensions, compute_risk_scorecard
from src.alert_engine import initialize_alert_store, add_alert, update_alert_status, get_prioritized_alerts
from src.analytics import compute_system_threat_level, compute_aggregate_metrics, compute_fraud_signal_board

# Load CSS Stylesheet
CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "custom.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "model_ref" not in st.session_state:
    try:
        st.session_state.model_ref = load_xgboost_model()
    except Exception as e:
        st.error(f"Failed to load XGBoost model: {e}")
        st.stop()

model = st.session_state.model_ref
initialize_alert_store()

# Pre-load sample benchmark data if history is empty
SAMPLE_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample", "sample_transactions.csv")
if not st.session_state.history and os.path.exists(SAMPLE_CSV_PATH):
    try:
        sample_df = pd.read_csv(SAMPLE_CSV_PATH)
        for _, row in sample_df.iterrows():
            prep_df = prepare_transaction_features(row.to_dict())
            res = analyze_transaction(model, prep_df, row.to_dict())
            res["raw_input"]["txn_id"] = row.get("txn_id", f"TXN-{len(st.session_state.history)+1001}")
            st.session_state.history.append(res)
    except Exception as e:
        pass


# ==========================================
# 1. TOP BRANDING HEADER & LIVE STATUS
# ==========================================
st.markdown("""
<div class="fx-brand-header">
    <div class="fx-logo-container">
        <div class="fx-logo-icon">FX</div>
        <div>
            <h1 class="fx-brand-title">FRAUDX — Financial Risk Intelligence</h1>
            <p class="fx-brand-subtitle">Detect. Explain. Prioritize. Investigate.</p>
        </div>
    </div>
    <div class="fx-status-bar">
        <div class="fx-status-item">
            <span class="fx-status-dot"></span>
            <span>XGBoost Engine: <strong>ONLINE</strong></span>
        </div>
        <div class="fx-status-item">
            <span class="fx-status-dot"></span>
            <span>Risk Engine: <strong>ONLINE</strong></span>
        </div>
        <div class="fx-status-item">
            <span class="fx-status-dot"></span>
            <span>SHAP Explainability: <strong>ENABLED</strong></span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### 🛡️ FRAUDX Operations")
    st.markdown("---")
    
    threat_info = compute_system_threat_level(st.session_state.history)
    st.markdown(f"**CURRENT THREAT LEVEL**")
    st.markdown(f'<div class="{threat_info["badge_class"]}">{threat_info["level"]}</div>', unsafe_allow_html=True)
    st.caption(threat_info["description"])
    st.markdown("---")
    
    st.markdown("#### ⚡ Quick Actions")
    if st.button("🔄 Reset Session Data", use_container_width=True):
        st.session_state.history = []
        st.session_state.alerts = []
        st.rerun()

    st.markdown("---")
    st.markdown("##### 📌 Model Specs")
    schema = get_model_schema(model)
    st.text(f"Engine: XGBoost Classifier")
    st.text(f"Features: {schema['n_features']} engineered")
    st.text(f"Classes: {schema['classes']}")


# ==========================================
# MAIN APPLICATION NAVIGATION TABS
# ==========================================
tabs = st.tabs([
    "📡 Threat Center",
    "🔍 Transaction Scanner",
    "🚨 Alert Queue",
    "🧪 What-If Simulator",
    "⚔️ Compare Transactions",
    "📊 Fraud Analytics",
    "🤖 Model Intelligence"
])


# ------------------------------------------
# TAB 1: THREAT CENTER & LIVE MONITOR
# ------------------------------------------
with tabs[0]:
    st.markdown("### 📡 Financial Risk Operations & Threat Center")
    st.caption("Real-time threat level assessment, key operational metrics, and simulated transaction stream monitoring.")
    
    metrics = compute_aggregate_metrics(st.session_state.history)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="fx-metric-card">
            <div class="fx-metric-label">TRANSACTIONS ANALYZED</div>
            <div class="fx-metric-value">{metrics['total_analyzed']}</div>
            <div class="fx-metric-subtext">Session Stream</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="fx-metric-card">
            <div class="fx-metric-label">FRAUD DETECTED</div>
            <div class="fx-metric-value score-high">{metrics['fraud_detected']}</div>
            <div class="fx-metric-subtext">{metrics['fraud_rate']}% Fraud Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="fx-metric-card">
            <div class="fx-metric-label">HIGH-RISK TRANSACTIONS</div>
            <div class="fx-metric-value score-high">{metrics['high_risk_count']}</div>
            <div class="fx-metric-subtext">Score &ge; 71</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="fx-metric-card">
            <div class="fx-metric-label">LEGITIMATE TRANSACTIONS</div>
            <div class="fx-metric-value score-low">{metrics['low_risk_count']}</div>
            <div class="fx-metric-subtext">Score &le; 30</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="fx-metric-card">
            <div class="fx-metric-label">AVERAGE RISK SCORE</div>
            <div class="fx-metric-value">{metrics['avg_risk_score']}</div>
            <div class="fx-metric-subtext">Scale 0 - 100</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.5, 1])
    
    with col_left:
        st.markdown("#### 📺 Live Transaction Feed")
        if st.session_state.history:
            feed_data = []
            for item in reversed(st.session_state.history[:10]):
                raw = item["raw_input"]
                feed_data.append({
                    "Txn ID": raw.get("txn_id", "TXN-XXXX"),
                    "Time": raw.get("timestamp", "09:42:00"),
                    "Type": raw.get("type", "TRANSFER"),
                    "Amount (₹)": f"₹{float(raw.get('amount', 0)):,.2f}",
                    "Risk Score": f"{item['risk_score']} / 100",
                    "Risk Level": item["risk_level"],
                    "Action": item["recommended_action"]
                })
            st.dataframe(pd.DataFrame(feed_data), use_container_width=True, hide_index=True)
        else:
            st.info("No transaction stream active. Use the Transaction Scanner tab to analyze transactions.")

    with col_right:
        st.markdown("#### 📡 Simulated Live Monitoring")
        st.caption("Process a batch of simulated transactions through the XGBoost engine.")
        
        if st.button("▶️ Start Live Stream Batch", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            simulated_batch = [
                {"txn_id": "TXN-SIM-01", "type": "TRANSFER", "amount": 450000.0, "oldbalanceOrg": 450000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "step": 12},
                {"txn_id": "TXN-SIM-02", "type": "PAYMENT", "amount": 65.0, "oldbalanceOrg": 3400.0, "newbalanceOrig": 3335.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "step": 12},
                {"txn_id": "TXN-SIM-03", "type": "CASH_OUT", "amount": 89000.0, "oldbalanceOrg": 89000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 12000.0, "newbalanceDest": 101000.0, "step": 12},
                {"txn_id": "TXN-SIM-04", "type": "TRANSFER", "amount": 1200.0, "oldbalanceOrg": 15000.0, "newbalanceOrig": 13800.0, "oldbalanceDest": 500.0, "newbalanceDest": 1700.0, "step": 12}
            ]
            
            for i, sim in enumerate(simulated_batch):
                status_text.text(f"Processing {sim['txn_id']} ({sim['type']} ₹{sim['amount']:,.2f})...")
                prep_df = prepare_transaction_features(sim)
                res = analyze_transaction(model, prep_df, sim)
                res["raw_input"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append(res)
                
                if res["risk_level"] in ["HIGH", "MEDIUM"]:
                    add_alert(res, sim["txn_id"])
                    
                time.sleep(0.3)
                progress_bar.progress((i + 1) / len(simulated_batch))
                
            status_text.success("Live stream batch processing complete!")
            st.rerun()


# ------------------------------------------
# TAB 2: TRANSACTION SCANNER & INVESTIGATION
# ------------------------------------------
with tabs[1]:
    st.markdown("### 🔍 Transaction Risk Scanner & Investigation Engine")
    st.caption("Input transaction parameters to run model inference, SHAP attribution, and full risk investigation.")

    with st.form("scanner_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"], index=0)
            amount = st.number_input("Transaction Amount (₹)", min_value=1.0, value=181000.0, step=1000.0)
            step = st.number_input("Simulation Step / Hour", min_value=1, max_value=744, value=14)
        with col_b:
            old_bal_org = st.number_input("Sender Old Balance (oldbalanceOrg)", min_value=0.0, value=181000.0, step=1000.0)
            new_bal_org = st.number_input("Sender New Balance (newbalanceOrig)", min_value=0.0, value=0.0, step=1000.0)
        with col_c:
            old_bal_dest = st.number_input("Receiver Old Balance (oldbalanceDest)", min_value=0.0, value=0.0, step=1000.0)
            new_bal_dest = st.number_input("Receiver New Balance (newbalanceDest)", min_value=0.0, value=0.0, step=1000.0)
            
        submitted = st.form_submit_button("⚡ ANALYZE TRANSACTION WITH XGBOOST", use_container_width=True)

    if submitted or "last_scan" in st.session_state:
        if submitted:
            raw_data = {
                "txn_id": f"TXN-{len(st.session_state.history)+1042}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": txn_type,
                "amount": amount,
                "oldbalanceOrg": old_bal_org,
                "newbalanceOrig": new_bal_org,
                "oldbalanceDest": old_bal_dest,
                "newbalanceDest": new_bal_dest,
                "step": step
            }
            prep_df = prepare_transaction_features(raw_data)
            assessment = analyze_transaction(model, prep_df, raw_data)
            st.session_state.last_scan = assessment
            st.session_state.history.append(assessment)
            
            if assessment["risk_level"] in ["HIGH", "MEDIUM"]:
                add_alert(assessment, raw_data["txn_id"])
        else:
            assessment = st.session_state.last_scan

        raw = assessment["raw_input"]
        shap_res = assessment["shap_analysis"]

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"## 🛡️ TRANSACTION INVESTIGATION: `{raw.get('txn_id', 'TXN-1042')}`")
        
        # Risk Badge & Score Banner
        res_col1, res_col2, res_col3 = st.columns([1, 1.2, 1.5])
        
        with res_col1:
            score_class = "score-high" if assessment['risk_score'] >= 71 else ("score-med" if assessment['risk_score'] >= 31 else "score-low")
            st.markdown(f"""
            <div class="fx-risk-box">
                <div class="fx-metric-label">OVERALL RISK SCORE</div>
                <div class="fx-risk-score-large {score_class}">{assessment['risk_score']}</div>
                <div style="font-family:var(--font-mono); font-weight:700;">{assessment['risk_level']} RISK</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="fx-risk-box">
                <div class="fx-metric-label">MODEL FRAUD PROBABILITY</div>
                <div class="fx-risk-score-large" style="font-size:2.5rem; color:var(--accent-sky);">{round(assessment['fraud_probability']*100, 1)}%</div>
                <div style="font-size:0.8rem; color:var(--text-secondary);">predict_proba() Result</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="fx-risk-box" style="text-align:left;">
                <div class="fx-metric-label">RECOMMENDED OPERATIONAL ACTION</div>
                <div style="font-size:1.25rem; font-weight:700; color:var(--text-primary); margin:0.4rem 0;">{assessment['recommended_action']}</div>
                <div style="font-size:0.82rem; color:var(--text-secondary);">{assessment['action_detail']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # SHAP Waterfall & Risk Radar Section
        col_shap, col_radar = st.columns([1.5, 1])
        
        with col_shap:
            st.markdown("#### 🌳 Real SHAP Feature Contributions")
            st.caption("SHAP (SHapley Additive exPlanations) exact log-odds attributions for this transaction.")
            
            if shap_res.get("available", False):
                contribs = shap_res["sorted_contributions"]
                df_shap = pd.DataFrame(contribs)
                
                # Plotly Horizontal Bar Chart of SHAP Contributions
                fig = px.bar(
                    df_shap,
                    x="shap_value",
                    y="display_name",
                    orientation="h",
                    color="impact",
                    color_discrete_map={"FRAUD": "#EF4444", "LEGITIMATE": "#10B981"},
                    labels={"shap_value": "SHAP Contribution (Log-Odds)", "display_name": "Feature"},
                    title="Feature Influence on Model Prediction"
                )
                fig.update_layout(
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="#F8FAFC", family="Inter"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📖 HOW TO READ THIS SHAP EXPLANATION"):
                    st.markdown("""
                    - **Positive SHAP Value (+)** (Red): Pushes the model prediction **toward FRAUD**.
                    - **Negative SHAP Value (-)** (Green): Pushes the model prediction **toward LEGITIMATE**.
                    - **Larger Bar Width**: Represents a stronger relative influence on this specific prediction.
                    """)
            else:
                st.warning(shap_res.get("error", "SHAP attribution unavailable."))

        with col_radar:
            st.markdown("#### 🎯 Transaction Risk Radar Profile")
            st.caption("Multi-dimensional risk assessment derived from data and model signals.")
            
            radar_dims = compute_risk_radar_dimensions(assessment["features_df"], assessment["fraud_probability"])
            radar_df = pd.DataFrame(list(radar_dims.items()), columns=["Dimension", "Score"])
            
            fig_radar = px.line_polar(
                radar_df, r="Score", theta="Dimension", line_close=True,
                range_r=[0, 100], color_discrete_sequence=["#38BDF8"]
            )
            fig_radar.update_traces(fill="toself", fillcolor="rgba(56, 189, 248, 0.2)")
            fig_radar.update_layout(
                paper_bgcolor="#111827",
                polar=dict(
                    bgcolor="#0A0E17",
                    radialaxis=dict(visible=True, range=[0, 100], color="#64748B"),
                    angularaxis=dict(color="#F8FAFC")
                ),
                font=dict(color="#F8FAFC", family="Inter"),
                margin=dict(l=30, r=30, t=20, b=20),
                height=350
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Summary & Scorecard Section
        col_summary, col_scorecard = st.columns([1.5, 1])
        
        with col_summary:
            st.markdown("""
            <div class="fx-investigation-card">
                <div class="fx-panel-title">💡 FRAUD EXPLANATION SUMMARY</div>
            """, unsafe_allow_html=True)
            st.markdown(shap_res.get("narrative_summary", "No narrative available."))
            
            st.markdown("<br><strong>Top Contributing Fraud Signals:</strong>", unsafe_allow_html=True)
            top_f = shap_res.get("top_fraud_signals", [])
            if top_f:
                for idx, sig in enumerate(top_f, 1):
                    st.markdown(f"{idx}. **{sig['display_name']}**: SHAP `+{sig['shap_value']:.4f}` (Value: `{sig['feature_value']}`)")
            else:
                st.markdown("No positive fraud signals identified.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_scorecard:
            scorecard = compute_risk_scorecard(assessment["features_df"], assessment["fraud_probability"], assessment["risk_score"])
            st.markdown("""
            <div class="fx-investigation-card">
                <div class="fx-panel-title">📋 RISK ASSESSMENT SCORECARD</div>
            """, unsafe_allow_html=True)
            for k, v in scorecard.items():
                st.markdown(f"**{k}**: `<span style='color:var(--accent-sky); font-family:var(--font-mono);'>{v}</span>`", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------
# TAB 3: ALERT PRIORITY QUEUE
# ------------------------------------------
with tabs[2]:
    st.markdown("### 🚨 Fraud Alert Priority Queue")
    st.caption("Prioritized list of suspicious transactions requiring analyst verification.")
    
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        status_filter = st.selectbox("Filter Status", ["ALL", "NEW", "UNDER REVIEW", "RESOLVED"])
        
    prioritized_alerts = get_prioritized_alerts(status_filter)
    
    if prioritized_alerts:
        for alt in prioritized_alerts:
            p_class = "badge-critical" if alt["priority"] == "CRITICAL" else ("badge-elevated" if alt["priority"] == "HIGH" else "badge-normal")
            
            with st.expander(f"[{alt['priority']}] {alt['alert_id']} — {alt['txn_id']} | Risk Score: {alt['risk_score']} / 100 ({alt['status']})"):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.markdown(f"**Transaction ID**: `{alt['txn_id']}`")
                    st.markdown(f"**Type**: `{alt['type']}`")
                    st.markdown(f"**Amount**: `₹{alt['amount']:,.2f}`")
                with c2:
                    st.markdown(f"**Risk Score**: `{alt['risk_score']} / 100`")
                    st.markdown(f"**Fraud Proba**: `{round(alt['fraud_proba']*100,1)}%`")
                    st.markdown(f"**Primary Signal**: `{alt['primary_signal']}`")
                with c3:
                    st.markdown(f"**Current Status**: `{alt['status']}`")
                    new_st = st.selectbox(f"Update Status for {alt['alert_id']}", ["NEW", "UNDER REVIEW", "RESOLVED"], index=["NEW", "UNDER REVIEW", "RESOLVED"].index(alt['status']), key=f"sel_{alt['alert_id']}")
                    if new_st != alt['status']:
                        update_alert_status(alt['alert_id'], new_st)
                        st.rerun()

                st.markdown("---")
                st.markdown("##### ⏱️ Investigation Timeline")
                for t_item in alt["timeline"]:
                    st.markdown(f"• `{t_item['time']}` — {t_item['event']}")
    else:
        st.info("No alerts match the selected status filter.")


# ------------------------------------------
# TAB 4: WHAT-IF RISK SIMULATOR
# ------------------------------------------
with tabs[3]:
    st.markdown("### 🧪 What-If Risk Simulator")
    st.caption("Adjust transaction inputs dynamically and observe how the XGBoost engine re-evaluates risk scores in real-time.")

    col_sim_in, col_sim_out = st.columns([1, 1])
    
    with col_sim_in:
        st.markdown("#### ⚙️ Adjust Parameters")
        sim_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT"], index=0, key="sim_type")
        sim_amt = st.slider("Transaction Amount (₹)", min_value=1000.0, max_value=2000000.0, value=50000.0, step=10000.0)
        sim_old_org = st.number_input("Sender Old Balance", min_value=0.0, value=50000.0, key="sim_old_org")
        sim_wipe = st.checkbox("Wipe Origin Balance to Zero", value=True)
        sim_new_org = 0.0 if sim_wipe else max(0.0, sim_old_org - sim_amt)
        
        sim_raw = {
            "type": sim_type,
            "amount": sim_amt,
            "oldbalanceOrg": sim_old_org,
            "newbalanceOrig": sim_new_org,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "step": 14
        }
        sim_df = prepare_transaction_features(sim_raw)
        sim_res = analyze_transaction(model, sim_df, sim_raw)

    with col_sim_out:
        st.markdown("#### 🎯 Simulated Model Output")
        st.markdown(f"""
        <div class="fx-risk-box">
            <div class="fx-metric-label">SIMULATED RISK SCORE</div>
            <div class="fx-risk-score-large {'score-high' if sim_res['risk_score']>=71 else 'score-low'}">{sim_res['risk_score']} / 100</div>
            <div style="font-weight:700;">{sim_res['recommended_action']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### SHAP Top Contributing Signal")
        top_f = sim_res["shap_analysis"].get("top_fraud_signals", [])
        if top_f:
            st.warning(f"Primary Fraud Signal: **{top_f[0]['display_name']}** (+{top_f[0]['shap_value']:.3f})")
        else:
            st.success("No strong positive fraud signals detected.")


# ------------------------------------------
# TAB 5: COMPARE TRANSACTIONS
# ------------------------------------------
with tabs[4]:
    st.markdown("### ⚔️ Side-by-Side Transaction Comparison")
    st.caption("Compare 2 transactions directly to understand differential SHAP attributions.")
    
    if len(st.session_state.history) >= 2:
        col_c1, col_c2 = st.columns(2)
        txn_list = [f"{item['raw_input'].get('txn_id','TXN')} ({item['risk_score']} pts)" for item in st.session_state.history]
        
        with col_c1:
            idx1 = st.selectbox("Select Transaction A", range(len(txn_list)), index=0)
            res1 = st.session_state.history[idx1]
        with col_c2:
            idx2 = st.selectbox("Select Transaction B", range(len(txn_list)), index=min(1, len(txn_list)-1))
            res2 = st.session_state.history[idx2]

        comp_data = {
            "Metric": ["Txn ID", "Type", "Amount (₹)", "Risk Score", "Fraud Proba", "Risk Level", "Action"],
            "Transaction A": [
                str(res1["raw_input"].get("txn_id")),
                str(res1["raw_input"].get("type")),
                f"₹{float(res1['raw_input'].get('amount', 0)):,.2f}",
                str(res1["risk_score"]),
                f"{round(float(res1['fraud_probability'])*100,1)}%",
                str(res1["risk_level"]),
                str(res1["recommended_action"])
            ],
            "Transaction B": [
                str(res2["raw_input"].get("txn_id")),
                str(res2["raw_input"].get("type")),
                f"₹{float(res2['raw_input'].get('amount', 0)):,.2f}",
                str(res2["risk_score"]),
                f"{round(float(res2['fraud_probability'])*100,1)}%",
                str(res2["risk_level"]),
                str(res2["recommended_action"])
            ]
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    else:
        st.info("Analyze at least 2 transactions to enable side-by-side comparison.")


# ------------------------------------------
# TAB 6: FRAUD ANALYTICS
# ------------------------------------------
with tabs[5]:
    st.markdown("### 📊 Operational Analytics & Global SHAP Insights")
    st.caption("Aggregate fraud patterns, risk distributions, and global feature importance across analyzed data.")
    
    if st.session_state.history:
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### Risk Level Distribution")
            df_hist = pd.DataFrame([{"risk_level": r["risk_level"]} for r in st.session_state.history])
            fig_pie = px.pie(
                df_hist, names="risk_level",
                color="risk_level",
                color_discrete_map={"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}
            )
            fig_pie.update_layout(paper_bgcolor="#111827", font=dict(color="#F8FAFC"))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_g2:
            st.markdown("#### 🔥 Top Fraud Signal Board")
            signal_board = compute_fraud_signal_board(st.session_state.history)
            if signal_board:
                st.dataframe(pd.DataFrame(signal_board), use_container_width=True, hide_index=True)
            else:
                st.info("No high-risk fraud signals recorded yet.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("#### 🌐 Global SHAP Feature Importance")
        st.caption("Calculates mean absolute SHAP values across all session transactions.")
        
        all_features = pd.concat([r["features_df"] for r in st.session_state.history], ignore_index=True)
        global_shap = get_global_shap_importance(model, all_features)
        
        if not global_shap.empty:
            fig_glob = px.bar(
                global_shap, x="importance", y="display_name", orientation="h",
                color_discrete_sequence=["#38BDF8"],
                labels={"importance": "Mean |SHAP Value|", "display_name": "Feature"}
            )
            fig_glob.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(color="#F8FAFC"))
            st.plotly_chart(fig_glob, use_container_width=True)
    else:
        st.info("No transaction history available for analytics.")


# ------------------------------------------
# TAB 7: MODEL INTELLIGENCE & RESPONSIBLE AI
# ------------------------------------------
with tabs[6]:
    st.markdown("### 🤖 Model Intelligence & Responsible AI")
    st.caption("Technical model specifications, architectural decisions, and AI safety disclosures.")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("""
        <div class="fx-investigation-card">
            <div class="fx-panel-title">🧠 MODEL ARCHITECTURE</div>
            <ul>
                <li><strong>Algorithm:</strong> XGBoost (Extreme Gradient Boosting) Classifier</li>
                <li><strong>Task:</strong> Binary Tabular Classification (0: Legitimate, 1: Fraudulent)</li>
                <li><strong>Engine Input:</strong> 11 preprocessed & engineered features</li>
                <li><strong>Explainability Engine:</strong> Genuine SHAP TreeExplainer</li>
                <li><strong>Inference Output:</strong> predict_proba() Fraud Probability + Normalized 0-100 Risk Score</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="fx-investigation-card">
            <div class="fx-panel-title">⚖️ RESPONSIBLE AI & LIMITATIONS</div>
            <ul>
                <li><strong>Decision Support Role:</strong> This system serves as a decision support prototype for anti-fraud ops teams.</li>
                <li><strong>Probabilistic Nature:</strong> Risk scores represent statistical probabilities, not absolute ground truth.</li>
                <li><strong>False Positive Management:</strong> Human analyst review remains vital for transactions flagged with medium/high risk scores.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
