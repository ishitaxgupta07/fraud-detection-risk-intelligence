"""Dashboard Page 1: Overview / KPIs."""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.config import Config

cfg = Config()


def _load_eda() -> dict:
    p = cfg.report_dir / "eda_report.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _load_model_comparison() -> pd.DataFrame:
    p = cfg.report_dir / "model_comparison.json"
    if p.exists():
        data = json.loads(p.read_text())
        return pd.DataFrame(data).T.reset_index().rename(columns={"index": "Model"})
    return pd.DataFrame()


def render():
    st.title("🛡️ Fraud Detection Dashboard")
    st.markdown("Real-time overview of the Explainable Hybrid Fraud Detection System.")
    st.markdown("---")

    eda = _load_eda()
    df_models = _load_model_comparison()

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    fraud_rate = eda.get("fraud_rate_pct", 1.8)
    total = eda.get("shape", [10000, 30])[0]
    dist = eda.get("class_distribution", {"0": 9800, "1": 200})
    fraud_count = dist.get("1", dist.get(1, 200))

    col1.metric("📊 Total Transactions", f"{total:,}")
    col2.metric("🚨 Fraud Cases", f"{fraud_count:,}")
    col3.metric("⚠️ Fraud Rate", f"{fraud_rate:.2f}%")
    col4.metric("✅ Best Model", "XGBoost" if df_models.empty else df_models.iloc[0]["Model"])

    st.markdown("---")

    # Class Distribution
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🍧 Class Distribution")
        labels = ["Normal", "Fraud"]
        values = [int(dist.get("0", dist.get(0, 9800))), int(fraud_count)]
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker_colors=["#2196F3", "#F44336"],
            )
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🏆 Model Comparison")
        if not df_models.empty:
            metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            available = [m for m in metrics if m in df_models.columns]
            fig2 = px.bar(
                df_models.melt(id_vars="Model", value_vars=available),
                x="variable",
                y="value",
                color="Model",
                barmode="group",
                labels={"variable": "Metric", "value": "Score"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(margin=dict(t=20, b=20), yaxis_range=[0, 1])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run training to see model comparison.")

    # Risk Score Gauge (demo)
    st.markdown("---")
    st.subheader("🎯 Live Risk Score Gauge (Demo)")
    demo_score = st.slider("Simulate Risk Score", 0, 100, 45)
    color = (
        "#28a745" if demo_score < 25 else "#ffc107" if demo_score < 50 else "#fd7e14" if demo_score < 75 else "#dc3545"
    )
    fig3 = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=demo_score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 25], "color": "#d4edda"},
                    {"range": [25, 50], "color": "#fff3cd"},
                    {"range": [50, 75], "color": "#ffe5d0"},
                    {"range": [75, 100], "color": "#f8d7da"},
                ],
            },
            title={"text": "Risk Score"},
        )
    )
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)
