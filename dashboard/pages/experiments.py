"""Dashboard Page 4: Research Experiments."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.config import Config

cfg = Config()


def render():
    st.title("🧪 Research Experiments")
    st.markdown("Compare different algorithms, imbalance strategies, and anomaly thresholds.")
    st.markdown("---")

    p = cfg.experiment_dir / "research_results.json"
    if p.exists():
        df = pd.read_json(p)
        st.subheader("📊 All Experiment Results")
        st.dataframe(df, use_container_width=True)

        for exp_type in df["experiment"].unique():
            sub = df[df["experiment"] == exp_type]
            st.subheader(f"Experiment: {exp_type}")
            metrics = [c for c in ["f1_score", "roc_auc", "precision", "recall"] if c in sub.columns]
            fig = px.bar(
                sub.melt(id_vars="name", value_vars=metrics),
                x="name",
                y="value",
                color="variable",
                barmode="group",
                labels={"name": "Config", "value": "Score"},
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run experiments to see results here.")
        _demo_experiments()

    rp = cfg.experiment_dir / "research_report.md"
    if rp.exists():
        st.markdown("---")
        st.subheader("📝 Research Report")
        st.markdown(rp.read_text())


def _demo_experiments():
    demo = pd.DataFrame(
        {
            "name": ["XGBoost_shallow", "XGBoost_deep", "RF_100", "RF_300", "XGB_smote", "XGB_undersample"],
            "f1_score": [0.82, 0.91, 0.78, 0.85, 0.89, 0.76],
            "roc_auc": [0.95, 0.98, 0.92, 0.96, 0.97, 0.90],
        }
    )
    fig = px.bar(
        demo.melt(id_vars="name", value_vars=["f1_score", "roc_auc"]),
        x="name",
        y="value",
        color="variable",
        barmode="group",
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Demo data - run experiments to see real results.")
