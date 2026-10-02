"""Dashboard Page 5: Analytics."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.config import Config

cfg = Config()


def render():
    st.title("📈 Analytics")
    st.markdown("Deep-dive analytics on transaction patterns and fraud trends.")
    st.markdown("---")

    # Simulated time-series fraud trend
    st.subheader("📅 Fraud Trend Over Time (Demo)")
    np.random.seed(42)
    days = pd.date_range("2024-01-01", periods=90)
    fraud_counts = np.random.poisson(5, 90) + np.sin(np.linspace(0, 6, 90)) * 3
    df_trend = pd.DataFrame({"Date": days, "Fraud Cases": fraud_counts.clip(0)})
    fig1 = px.line(df_trend, x="Date", y="Fraud Cases", markers=True, color_discrete_sequence=["#F44336"])
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Fraud by Amount Range")
        bins = ["<$100", "$100-$500", "$500-$1K", "$1K-$5K", ">$5K"]
        counts = [120, 85, 60, 40, 95]
        fig2 = px.bar(
            x=bins,
            y=counts,
            labels={"x": "Amount Range", "y": "Fraud Count"},
            color=counts,
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("🕒 Fraud by Hour of Day")
        hours = list(range(24))
        fraud_by_hour = np.random.poisson(3, 24)
        fraud_by_hour[0:5] += 8  # night spike
        fraud_by_hour[22:24] += 6
        fig3 = px.bar(
            x=hours,
            y=fraud_by_hour,
            labels={"x": "Hour", "y": "Fraud Count"},
            color=fraud_by_hour,
            color_continuous_scale="OrRd",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Risk category distribution
    st.subheader("🎯 Risk Category Distribution")
    cats = ["Low", "Medium", "High", "Critical"]
    vals = [6500, 2200, 900, 400]
    colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545"]
    fig4 = go.Figure(go.Bar(x=cats, y=vals, marker_color=colors))
    fig4.update_layout(xaxis_title="Risk Category", yaxis_title="Transaction Count")
    st.plotly_chart(fig4, use_container_width=True)
