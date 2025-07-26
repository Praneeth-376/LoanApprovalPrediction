import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter details to check if the loan will be approved")

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

if st.button("Predict Loan Status"):
    input_data = np.array([
        le.transform([Gender])[0],
        le.transform([Married])[0],
        le.transform([Dependents])[0],
        le.transform([Education])[0],
        le.transform([Self_Employed])[0],
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        le.transform([Property_Area])[0]
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
