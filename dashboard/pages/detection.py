"""Dashboard Page 2: Fraud Detection."""

import json

import numpy as np
import requests
import streamlit as st

from utils.config import Config

cfg = Config()
API_URL = "http://localhost:8000"


def render():
    st.title("🚨 Fraud Detection")
    st.markdown("Submit a transaction to get real-time fraud prediction.")
    st.markdown("---")

    mode = st.radio("Input Mode", ["Manual Feature Input", "Random Sample"])

    if mode == "Random Sample":
        np.random.seed(None)
        n_features = st.slider("Number of features", 10, 30, 28)
        features = np.random.randn(n_features).tolist()
        st.code(json.dumps([round(f, 4) for f in features[:5]]) + " ... (truncated)")
    else:
        raw = st.text_area(
            "Enter comma-separated feature values:",
            value="0.1, -1.2, 0.5, 2.3, -0.8, 1.1, 0.3, -0.5, 0.9, 1.5, "
            "-0.2, 0.7, -1.0, 0.4, 0.6, -0.3, 1.2, -0.9, 0.8, 0.1, "
            "-0.6, 0.2, -0.4, 1.3, -0.7, 0.5, 0.0, 150.0",
        )
        try:
            features = [float(x.strip()) for x in raw.split(",")]
        except ValueError:
            st.error("Invalid input. Please enter comma-separated numbers.")
            return

    if st.button("🔍 Predict", type="primary"):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=10,
            )
            if resp.status_code == 200:
                result = resp.json()
                _display_result(result)
            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.warning("⚠️ API not running. Showing demo result.")
            _display_result(
                {
                    "fraud_probability": 0.87,
                    "is_fraud": True,
                    "risk_score": 82.5,
                    "risk_category": "Critical",
                    "anomaly_score": 0.74,
                }
            )


def _display_result(result: dict):
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    fp = result["fraud_probability"]
    rs = result["risk_score"]
    cat = result["risk_category"]
    color_map = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
    col1.metric("📊 Fraud Probability", f"{fp*100:.1f}%")
    col2.metric("⚠️ Risk Score", f"{rs:.1f}/100")
    col3.metric(f"{color_map.get(cat, '')} Risk Category", cat)

    verdict = "🚨 FRAUD DETECTED" if result["is_fraud"] else "✅ LEGITIMATE TRANSACTION"
    color = "#dc3545" if result["is_fraud"] else "#28a745"
    st.markdown(
        f"<div style='background:{color};color:white;padding:1rem;border-radius:8px;"
        f"text-align:center;font-size:1.4rem;font-weight:bold;'>{verdict}</div>",
        unsafe_allow_html=True,
    )
