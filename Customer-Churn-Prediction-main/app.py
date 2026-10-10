import streamlit as st
import pandas as pd
from src.model_trainer import ChurnPredictorModel

st.set_page_config(page_title="📊 Customer Churn Prediction & Retention System", layout="wide", page_icon="📉")

st.title("📊 Customer Churn Prediction & Retention Analytics")
st.markdown("End-to-End Telecom Customer Retention Machine Learning Web Application inspired by **CampusX ML Projects**.")

@st.cache_resource
def load_churn_model():
    return ChurnPredictorModel()

model_engine = load_churn_model()

st.subheader("👤 Customer Demographic & Account Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col3:
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 18.0, 10000.0, tenure * monthly_charges)
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

if st.button("🚀 Predict Customer Churn Risk", type="primary"):
    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': internet_service,
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': tech_support,
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    result = model_engine.predict_single_customer(customer_data)
    
    st.markdown("---")
    st.subheader("🎯 Prediction Output & Risk Assessment")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Churn Risk Probability", f"{result['churn_probability']}%")
    with m2:
        st.metric("Model Classification Status", result['prediction'])
    with m3:
        st.metric("Model Training Accuracy", f"{result['model_accuracy']}%")
        
    st.progress(result['churn_probability'] / 100)
    
    if result['churn_probability'] >= 50:
        st.error("⚠️ HIGH CHURN RISK: Recommend proactive retention offer (e.g. 15% annual discount or upgrade to 1-Year Contract).")
    else:
        st.success("✅ LOW CHURN RISK: Customer demonstrates strong retention affinity.")
