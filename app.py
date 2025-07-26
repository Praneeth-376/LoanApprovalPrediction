
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and transformers
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("le_gender.pkl")
le_married = joblib.load("le_married.pkl")
le_dependents = joblib.load("le_dependents.pkl")
le_education = joblib.load("le_education.pkl")
le_self_employed = joblib.load("le_self_employed.pkl")
le_property_area = joblib.load("le_property_area.pkl")
feature_names = joblib.load("model_features.pkl")

# App UI
st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter applicant details to check loan approval status")

# User Inputs
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

# On Predict
if st.button("Predict Loan Status"):
    # Encode categorical values
    data_dict = {
        'Gender': le_gender.transform([Gender])[0],
        'Married': le_married.transform([Married])[0],
        'Dependents': le_dependents.transform([Dependents])[0],
        'Education': le_education.transform([Education])[0],
        'Self_Employed': le_self_employed.transform([Self_Employed])[0],
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': le_property_area.transform([Property_Area])[0],
    }

    input_df = pd.DataFrame([data_dict])[feature_names]  # reorder to match training

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
