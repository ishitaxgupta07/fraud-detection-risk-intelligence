"""Dashboard Page 3: SHAP Explainability."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.config import Config

cfg = Config()


def render():
    st.title("📊 SHAP Explainability")
    st.markdown("Understand **why** the model makes each prediction.")
    st.markdown("---")

    # Global importance
    p = cfg.report_dir / "global_feature_importance.json"
    if p.exists():
        df = pd.read_json(p)
        st.subheader("🌏 Global Feature Importance (Mean |SHAP|)")
        fig = px.bar(
            df.head(20),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="RdYlGn_r",
            labels={"importance": "Mean |SHAP|", "feature": "Feature"},
        )
        fig.update_layout(yaxis={"autorange": "reversed"}, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run training + SHAP analysis to see feature importance.")
        _demo_shap()

    # SHAP report
    rp = cfg.report_dir / "shap_report.md"
    if rp.exists():
        st.markdown("---")
        st.subheader("📝 SHAP Report")
        st.markdown(rp.read_text())


def _demo_shap():
    st.subheader("📊 Demo: Feature Importance")
    demo = pd.DataFrame(
        {
            "feature": ["Amount", "V14", "V4", "V12", "V10", "V17", "V3", "V7", "V11", "V16"],
            "importance": [0.42, 0.38, 0.31, 0.28, 0.25, 0.22, 0.19, 0.17, 0.14, 0.11],
        }
    )
    fig = px.bar(
        demo,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(yaxis={"autorange": "reversed"}, height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Demo data - run training to see real SHAP values.")
