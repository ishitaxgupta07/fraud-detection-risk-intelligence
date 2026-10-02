"""Phase 8: Streamlit Dashboard - Main Entry Point."""

import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f77b4; }
    .kpi-card { background: #f0f2f6; border-radius: 10px; padding: 1rem; text-align: center; }
    .risk-low    { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high   { color: #fd7e14; font-weight: bold; }
    .risk-critical { color: #dc3545; font-weight: bold; }
    .stMetric label { font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='main-header'>🛡️ FraudGuard</div>", unsafe_allow_html=True)
    st.markdown("---")
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Fraud Detection", "SHAP Explainability", "Research Experiments", "Analytics"],
        icons=["speedometer2", "shield-exclamation", "bar-chart-line", "flask", "graph-up"],
        default_index=0,
    )

if selected == "Dashboard":
    from dashboard.pages.overview import render
elif selected == "Fraud Detection":
    from dashboard.pages.detection import render
elif selected == "SHAP Explainability":
    from dashboard.pages.shap_page import render
elif selected == "Research Experiments":
    from dashboard.pages.experiments import render
else:
    from dashboard.pages.analytics import render

render()
