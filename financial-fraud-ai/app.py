import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="Financial Fraud Detection AI",
    page_icon="💰",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================

model = joblib.load("model/fraud_model.pkl")

# ======================
# CUSTOM CSS
# ======================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3 {
    color: #38bdf8;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    border: none;
}

[data-testid="metric-container"] {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# ======================
# TITLE
# ======================

st.title("💰 Financial Statement Fraud Detection AI")

st.write("""
AI system for detecting fraud risk in financial statements.
Enter financial indicators below to predict whether a company is suspicious.
""")

# ======================
# INPUT SECTION
# ======================

st.subheader("📊 Enter Financial Information")

col1, col2 = st.columns(2)

with col1:

    total_assets = st.number_input(
        "Total Assets",
        min_value=0.0,
        value=1000000.0
    )

    total_liabilities = st.number_input(
        "Total Liabilities",
        min_value=0.0,
        value=500000.0
    )

    revenue = st.number_input(
        "Revenue",
        min_value=0.0,
        value=750000.0
    )

    operating_expenses = st.number_input(
        "Operating Expenses",
        min_value=0.0,
        value=400000.0
    )

    net_income = st.number_input(
        "Net Income",
        value=100000.0
    )

    cash_flow_operating = st.number_input(
        "Cash Flow Operating",
        value=120000.0
    )

    cash_flow_investing = st.number_input(
        "Cash Flow Investing",
        value=-50000.0
    )

with col2:

    cash_flow_financing = st.number_input(
        "Cash Flow Financing",
        value=20000.0
    )

    current_ratio = st.number_input(
        "Current Ratio",
        min_value=0.0,
        value=1.5
    )

    debt_to_equity = st.number_input(
        "Debt to Equity",
        min_value=0.0,
        value=0.8
    )

    gross_margin = st.number_input(
        "Gross Margin",
        value=35.0
    )

    return_on_assets = st.number_input(
        "Return on Assets",
        value=5.0
    )

    return_on_equity = st.number_input(
        "Return on Equity",
        value=10.0
    )

# ======================
# PREDICT BUTTON
# ======================

if st.button("🔍 Predict Fraud Risk"):

    # ======================
    # CREATE DATAFRAME
    # ======================

    input_data = pd.DataFrame({
        "Total_Assets": [total_assets],
        "Total_Liabilities": [total_liabilities],
        "Revenue": [revenue],
        "Operating_Expenses": [operating_expenses],
        "Net_Income": [net_income],
        "Cash_Flow_Operating": [cash_flow_operating],
        "Cash_Flow_Investing": [cash_flow_investing],
        "Cash_Flow_Financing": [cash_flow_financing],
        "Current_Ratio": [current_ratio],
        "Debt_to_Equity": [debt_to_equity],
        "Gross_Margin": [gross_margin],
        "Return_on_Assets": [return_on_assets],
        "Return_on_Equity": [return_on_equity]
    })

    # ======================
    # PREDICT
    # ======================

    prediction = model.predict(input_data)[0]

    # ======================
    # RESULT
    # ======================

    st.subheader("📌 Prediction Result")

    risk_score = 0

if debt_to_equity > 10:
    risk_score += 1

if current_ratio < 0.5:
    risk_score += 1

if net_income > revenue:
    risk_score += 1

if gross_margin > 80:
    risk_score += 1

if risk_score >= 2:

    st.error("⚠ HIGH FRAUD RISK DETECTED")

else:

    st.success("✅ Financial Statement Looks Normal")

    # ======================
    # CHART
    # ======================

    st.subheader("📈 Financial Overview")

    chart_df = pd.DataFrame({
        "Metric": [
            "Assets",
            "Liabilities",
            "Revenue",
            "Expenses",
            "Net Income"
        ],
        "Value": [
            total_assets,
            total_liabilities,
            revenue,
            operating_expenses,
            net_income
        ]
    })

    fig = px.bar(
        chart_df,
        x="Metric",
        y="Value",
        template="plotly_dark",
        title="Financial Summary"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ======================
# FOOTER
# ======================

st.markdown("---")

st.caption("AI Financial Fraud Detection System | Streamlit + Machine Learning")