import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, label encoder, and feature names
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
feature_names = joblib.load('model_features.pkl')

st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter the applicant details below to check whether the loan is likely to be approved.")

# Input fields
Gender = st.selectbox("Gender", ['Male', 'Female'])
Married = st.selectbox("Married", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
ApplicantIncome = st.number_input("Applicant Income", value=5000)
CoapplicantIncome = st.number_input("Coapplicant Income", value=0)
LoanAmount = st.number_input("Loan Amount (in thousands)", value=150)
Loan_Amount_Term = st.selectbox("Loan Amount Term", [360.0, 120.0, 240.0, 180.0])
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

# Prediction
if st.button("Predict Loan Status"):
    # Create dictionary of features
    data_dict = {
        'Gender': le.transform([Gender])[0],
        'Married': le.transform([Married])[0],
        'Dependents': le.transform([Dependents])[0],
        'Education': le.transform([Education])[0],
        'Self_Employed': le.transform([Self_Employed])[0],
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': le.transform([Property_Area])[0]
    }

    # Create DataFrame in correct column order
    input_df = pd.DataFrame([data_dict])[feature_names]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    # Output result
    if prediction[0] == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
