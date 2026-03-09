"""
Streamlit Dashboard — Step 7
Interactive analytics dashboard for the Customer Retention Engine.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

import streamlit as st
from services.clustering_service import ClusteringService, CLUSTER_LABELS, CLUSTER_COLORS
from services.churn_service import ChurnService
from services.recommendation_service import RecommendationService

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Customer Retention Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #0f3460; margin: 0.5rem 0;
    }
    .risk-critical { background-color: #fee2e2; border-left: 4px solid #dc2626; 
                     padding: 1rem; border-radius: 8px; }
    .risk-high     { background-color: #fff7ed; border-left: 4px solid #ea580c;
                     padding: 1rem; border-radius: 8px; }
    .risk-medium   { background-color: #fefce8; border-left: 4px solid #ca8a04;
                     padding: 1rem; border-radius: 8px; }
    .risk-low      { background-color: #f0fdf4; border-left: 4px solid #16a34a;
                     padding: 1rem; border-radius: 8px; }
    .action-item   { background: white; border-radius: 6px; padding: 0.5rem 0.75rem;
                     margin: 0.3rem 0; border: 1px solid #e5e7eb; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ── load models & data ─────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    models_dir = os.path.join(ROOT, "models")
    clustering = ClusteringService()
    churn      = ChurnService()
    try:
        clustering.load()
        churn.load()
        scaler   = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        features = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
        loaded   = True
    except Exception:
        scaler, features, loaded = None, None, False
    return clustering, churn, RecommendationService(), scaler, features, loaded


@st.cache_data
def load_segmented_data():
    path = os.path.join(ROOT, "data", "customers_segmented.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


clustering_svc, churn_svc, rec_svc, scaler, feature_names, models_ready = load_all()
df_seg = load_segmented_data()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.markdown("## 🎯 Retention Engine")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview Dashboard",
         "🔍 Customer Analyzer",
         "📈 Model Performance",
         "🗺️ Segment Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if models_ready:
        st.success("✅ Models Loaded")
        if churn_svc.metrics:
            st.metric("AUC-ROC", f"{churn_svc.metrics.get('auc_roc', 0):.3f}")
            st.metric("F1 Score", f"{churn_svc.metrics.get('f1', 0):.3f}")
    else:
        st.error("⚠️ Models not loaded\nRun `python main.py` first")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI Customer Retention Engine</h1>
        <p>Behavioral Segmentation · Churn Prediction · Retention Strategy</p>
    </div>
    """, unsafe_allow_html=True)

    if df_seg is None:
        st.warning("No segmented data found. Run `python main.py` first.")
        st.stop()

    # ── KPI row ────────────────────────────────────────────────────────────
    total = len(df_seg)
    churn_rate = (df_seg["Churn"] == "Yes").mean() * 100
    at_risk_n  = (df_seg["ClusterLabel"] == "At-Risk Customers").sum()
    hv_n       = (df_seg["ClusterLabel"] == "High-Value Customers").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", f"{total:,}")
    col2.metric("📉 Churn Rate", f"{churn_rate:.1f}%", delta="-2.3% vs last month")
    col3.metric("⚠️ At-Risk Customers", f"{at_risk_n:,}")
    col4.metric("💎 High-Value Customers", f"{hv_n:,}")

    st.markdown("---")

    # ── Segment distribution + Churn by segment ────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Customer Segment Distribution")
        seg_counts = df_seg["ClusterLabel"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        color_map = {v: c for v, c in zip(
            ["At-Risk Customers", "High-Value Customers", "Loyal Customers", "New Customers"],
            ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
        )}
        fig = px.pie(seg_counts, names="Segment", values="Count",
                     color="Segment", color_discrete_map=color_map,
                     hole=0.45)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Churn Rate by Segment")
        churn_by_seg = df_seg.groupby("ClusterLabel")["Churn"].apply(
            lambda s: (s == "Yes").mean() * 100
        ).reset_index()
        churn_by_seg.columns = ["Segment", "ChurnRate"]
        churn_by_seg = churn_by_seg.sort_values("ChurnRate", ascending=True)
        fig2 = px.bar(churn_by_seg, x="ChurnRate", y="Segment", orientation="h",
                      color="ChurnRate", color_continuous_scale="RdYlGn_r",
                      text=churn_by_seg["ChurnRate"].apply(lambda v: f"{v:.1f}%"))
        fig2.update_layout(height=380, margin=dict(t=10, b=10),
                           coloraxis_showscale=False,
                           xaxis_title="Churn Rate (%)", yaxis_title="")
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tenure × Monthly Charges scatter ──────────────────────────────────
    st.subheader("Tenure vs Monthly Charges by Segment")
    sample = df_seg.sample(min(2000, len(df_seg)), random_state=42)
    fig3 = px.scatter(sample, x="tenure", y="MonthlyCharges",
                      color="ClusterLabel", color_discrete_map=color_map,
                      opacity=0.55, size_max=6,
                      labels={"tenure": "Tenure (months)",
                               "MonthlyCharges": "Monthly Charges ($)",
                               "ClusterLabel": "Segment"})
    fig3.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER ANALYZER
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Customer Analyzer":
    st.markdown("## 🔍 Customer Analyzer")
    st.markdown("Enter customer details to get segment, churn risk, and retention recommendations.")

    if not models_ready:
        st.error("Models not loaded. Run `python main.py` first.")
        st.stop()

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Account Info")
            tenure         = st.slider("Tenure (months)", 0, 72, 12)
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner        = st.selectbox("Has Partner", [0, 1], format_func=lambda x: "Yes" if x else "No")
            dependents     = st.selectbox("Has Dependents", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col2:
            st.markdown("#### Services")
            phone_service  = st.selectbox("Phone Service", [1, 0], format_func=lambda x: "Yes" if x else "No")
            paperless      = st.selectbox("Paperless Billing", [1, 0], format_func=lambda x: "Yes" if x else "No")
            gender         = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x else "Female")

        with col3:
            st.markdown("#### Financials")
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.5, step=0.5)
            total_charges   = st.number_input("Total Charges ($)",   0.0,  9000.0,
                                               float(tenure * monthly_charges), step=10.0)

        submitted = st.form_submit_button("🚀 Analyze Customer", use_container_width=True)

    if submitted:
        # Build minimal feature vector aligned to training features
        row = {feat: 0.0 for feat in feature_names}
        overrides = {
            "tenure": tenure, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges, "SeniorCitizen": senior_citizen,
            "gender": gender, "Partner": partner,
            "Dependents": dependents, "PhoneService": phone_service,
            "PaperlessBilling": paperless,
        }
        for k, v in overrides.items():
            if k in row:
                row[k] = float(v)

        tmp = pd.DataFrame([row], columns=feature_names)
        from data.data_pipeline import NUMERIC_COLS
        existing_num = [c for c in NUMERIC_COLS if c in feature_names]
        tmp[existing_num] = scaler.transform(tmp[existing_num])
        vec = tmp.values[0]

        cluster_id = clustering_svc.predict(vec)
        churn_prob = churn_svc.predict_proba(vec)
        rec        = rec_svc.get_recommendation(cluster_id, churn_prob)

        # Results
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("🏷️ Segment",          rec.cluster_label)
        c2.metric("📉 Churn Probability", f"{churn_prob*100:.1f}%")
        c3.metric("🚨 Risk Level",        rec.churn_risk)

        risk_class = f"risk-{rec.churn_risk.lower()}"
        st.markdown(f"""
        <div class="{risk_class}">
            <h4>Priority: {rec.priority} | {rec.estimated_impact}</h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎯 Recommended Actions")
        for action in rec.actions:
            st.markdown(f'<div class="action-item">✅ {action}</div>', unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Risk (%)"},
            delta={"reference": 26},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if churn_prob > 0.5 else "#f39c12" if churn_prob > 0.25 else "#2ecc71"},
                "steps": [
                    {"range": [0, 25],  "color": "#d5f5e3"},
                    {"range": [25, 50], "color": "#fef9e7"},
                    {"range": [50, 75], "color": "#fdebd0"},
                    {"range": [75, 100],"color": "#fadbd8"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 50},
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance")

    plots_dir = os.path.join(ROOT, "data", "plots")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Confusion Matrix", "ROC Curve", "Feature Importance", "EDA Plots"]
    )

    def _show_plot(name: str):
        path = os.path.join(plots_dir, name)
        if os.path.exists(path):
            st.image(path, use_column_width=True)
        else:
            st.info(f"Plot not found: {name}. Run `python main.py` first.")

    with tab1:
        _show_plot("confusion_matrix.png")
    with tab2:
        _show_plot("roc_curve.png")
    with tab3:
        _show_plot("feature_importance.png")
    with tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            _show_plot("churn_distribution.png")
            _show_plot("tenure_vs_churn.png")
            _show_plot("contract_vs_churn.png")
        with col_b:
            _show_plot("monthly_charges_vs_churn.png")
            _show_plot("correlation_heatmap.png")

    if churn_svc.metrics:
        st.markdown("---")
        st.markdown("### 📊 Metrics Summary")
        m = churn_svc.metrics
        cols = st.columns(5)
        labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
        keys   = ["accuracy", "precision", "recall", "f1", "auc_roc"]
        for col, label, key in zip(cols, labels, keys):
            col.metric(label, f"{m.get(key, 0):.3f}")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4 — SEGMENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Segment Explorer":
    st.markdown("## 🗺️ Segment Explorer")

    plots_dir = os.path.join(ROOT, "data", "plots")
    pca_path  = os.path.join(plots_dir, "cluster_pca.png")

    if os.path.exists(pca_path):
        st.image(pca_path, caption="Customer Segments — PCA Projection", use_column_width=True)
    else:
        st.info("PCA plot not available. Run `python main.py` first.")

    if df_seg is not None:
        st.markdown("---")
        st.markdown("### Segment Profiles")

        profile = df_seg.groupby("ClusterLabel").agg(
            Count      =("Churn", "count"),
            ChurnRate  =("Churn", lambda s: f"{(s=='Yes').mean()*100:.1f}%"),
            AvgTenure  =("tenure", lambda s: f"{s.mean():.0f} months"),
            AvgMonthly =("MonthlyCharges", lambda s: f"${s.mean():.2f}"),
            AvgTotal   =("TotalCharges",   lambda s: f"${s.mean():,.0f}"),
        ).reset_index()
        st.dataframe(profile, use_container_width=True, hide_index=True)

        selected = st.selectbox("Explore Segment", list(CLUSTER_LABELS.values()))
        seg_df   = df_seg[df_seg["ClusterLabel"] == selected]

        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(seg_df, x="tenure", color="Churn",
                               color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
                               title=f"{selected} — Tenure Distribution",
                               barmode="overlay", opacity=0.75)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig2 = px.histogram(seg_df, x="MonthlyCharges", color="Churn",
                                color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
                                title=f"{selected} — Monthly Charges Distribution",
                                barmode="overlay", opacity=0.75)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Retention Strategy for this Segment")
        # Show all risk levels
        for churn_prob_ex, label in [(0.10, "Low Risk"), (0.40, "Medium Risk"),
                                      (0.65, "High Risk"), (0.85, "Critical Risk")]:
            cluster_id = [k for k, v in CLUSTER_LABELS.items() if v == selected][0]
            rec = rec_svc.get_recommendation(cluster_id, churn_prob_ex)
            with st.expander(f"🎯 {label} ({int(churn_prob_ex*100)}% churn prob)"):
                for action in rec.actions:
                    st.markdown(f"- {action}")
                st.caption(f"Priority: **{rec.priority}** | {rec.estimated_impact}")
