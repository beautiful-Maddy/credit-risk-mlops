import os
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.features.preprocessing import fit_scaler_and_columns


st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="✨",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #fff9fb 0%, #fffdfd 100%);
    }

    .main-title {
        text-align: center;
        color: #5c4b51;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: 0.5px;
    }

    .subtitle {
        text-align: center;
        color: #8c7b81;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 22px;
        box-shadow: 0 8px 24px rgba(220, 196, 204, 0.18);
        border: 1px solid #f3e4e8;
        margin-bottom: 1.2rem;
    }

    .result-card {
        background: #fffafb;
        padding: 1.2rem;
        border-radius: 20px;
        border: 1px solid #f2dfe5;
        box-shadow: 0 6px 18px rgba(220, 196, 204, 0.12);
    }

    .small-label {
        color: #9b8a90;
        font-size: 0.88rem;
        margin-bottom: 0.2rem;
    }

    .probability {
        color: #b76e79;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    div.stButton > button {
        background-color: #e8bfc8;
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 1.4rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #dca8b5;
        color: white;
    }

    div[data-baseweb="select"] > div {
        border-radius: 14px !important;
        border: 1px solid #ead7dd !important;
    }

    div[data-baseweb="input"] > div {
        border-radius: 14px !important;
        border: 1px solid #ead7dd !important;
    }

    .stNumberInput input {
        border-radius: 14px !important;
    }
    </style>
""", unsafe_allow_html=True)


model = joblib.load("models/logistic_model.pkl")
scaler, feature_columns = fit_scaler_and_columns()

st.markdown('<div class="main-title">Credit Risk Prediction ✨</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Client profile")

st.sidebar.markdown("### Model Info")
st.sidebar.write("Model: Logistic Regression")
st.sidebar.write("Recall: 0.67")

col1, col2 = st.columns(2)

with col1:
    credit_policy = st.selectbox("Credit Policy", [0, 1])
    purpose = st.selectbox(
        "Purpose",
        [
            "credit_card",
            "debt_consolidation",
            "educational",
            "home_improvement",
            "major_purchase",
            "small_business",
            "all_other",
        ],
    )
    int_rate = st.number_input("Interest Rate", min_value=0.0, value=0.12, step=0.01)
    installment = st.number_input("Installment", min_value=0.0, value=250.0, step=10.0)
    log_annual_inc = st.number_input("Log Annual Income", min_value=0.0, value=10.5, step=0.1)
    dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=12.0, step=0.1)

with col2:
    fico = st.number_input("FICO Score", min_value=300, max_value=850, value=700, step=1)
    days_with_cr_line = st.number_input("Days with Credit Line", min_value=0.0, value=4000.0, step=100.0)
    revol_bal = st.number_input("Revolving Balance", min_value=0, value=10000, step=100)
    revol_util = st.number_input("Revolving Utilization", min_value=0.0, value=45.0, step=0.1)
    inq_last_6mths = st.number_input("Inquiries Last 6 Months", min_value=0, value=1, step=1)
    delinq_2yrs = st.number_input("Delinquencies in 2 Years", min_value=0, value=0, step=1)
    pub_rec = st.number_input("Public Records", min_value=0, value=0, step=1)

predict_button = st.button("Predict risk")
st.markdown('</div>', unsafe_allow_html=True)

if predict_button:
    input_data = {
        "credit.policy": credit_policy,
        "int.rate": int_rate,
        "installment": installment,
        "log.annual.inc": log_annual_inc,
        "dti": dti,
        "fico": fico,
        "days.with.cr.line": days_with_cr_line,
        "revol.bal": revol_bal,
        "revol.util": revol_util,
        "inq.last.6mths": inq_last_6mths,
        "delinq.2yrs": delinq_2yrs,
        "pub.rec": pub_rec,
    }

    input_df = pd.DataFrame([input_data])

    for col in feature_columns:
        if col.startswith("purpose_"):
            input_df[col] = 0

    purpose_col = f"purpose_{purpose}"
    if purpose_col in input_df.columns:
        input_df[purpose_col] = 1

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-label">Default probability</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="probability">{probability:.2%}</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.error("High risk of default detected.")
    else:
        st.success("Low risk of default detected.")

    st.markdown('</div>', unsafe_allow_html=True)