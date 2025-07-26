# @title 5. Streamlit Application Code (app.py - Use this EXACT code)
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Pre-trained Model and Preprocessors ---
@st.cache_resource # Cache the loading of resources to avoid re-loading on every rerun
def load_resources():
    try:
        # Load each specific LabelEncoder
        le_gender = joblib.load('le_gender.pkl')
        le_married = joblib.load('le_married.pkl')
        le_dependents = joblib.load('le_dependents.pkl')
        le_education = joblib.load('le_education.pkl')
        le_self_employed = joblib.load('le_self_employed.pkl')
        le_property_area = joblib.load('le_property_area.pkl')
        
        # Load imputer, scaler, and model
        imputer = joblib.load('imputer.pkl')
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('model.pkl')
        
        return le_gender, le_married, le_dependents, le_education, le_self_employed, le_property_area, imputer, scaler, model
    except FileNotFoundError:
        st.error("Error: Model or preprocessor files not found. Please ensure Cell 3 (Train Model) was run successfully and all .pkl files were created.")
        st.stop() # Stop the app if files are missing

# Assign the loaded resources to their respective variables
le_gender, le_married, le_dependents, le_education, le_self_employed, le_property_area, imputer, scaler, model = load_resources()

# --- 2. Streamlit UI ---
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("🏡 Loan Approval Prediction App")
st.markdown("""
    Fill in the details below to predict if a loan will be approved.
""")

# Input fields for user
st.header("Applicant Information")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    married = st.selectbox("Married", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
    education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox("Self Employed", ['No', 'Yes'])
with col2:
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=120)
    loan_amount_term = st.selectbox("Loan Amount Term (Days)", [12, 36, 60, 120, 180, 240, 300, 360, 480], index=7)
    credit_history = st.selectbox("Credit History (1: Good, 0: Bad)", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

# --- 3. Prediction Logic ---
if st.button("Predict Loan Status"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    # Apply transformations in the same order as in training
    try:
        # Categorical features encoding using the LOADED, FITTED specific encoders
        input_data['Gender'] = le_gender.transform(input_data['Gender'])
        input_data['Married'] = le_married.transform(input_data['Married'])
        input_data['Dependents'] = le_dependents.transform(input_data['Dependents'])
        input_data['Education'] = le_education.transform(input_data['Education'])
        input_data['Self_Employed'] = le_self_employed.transform(input_data['Self_Employed'])
        input_data['Property_Area'] = le_property_area.transform(input_data['Property_Area'])

        # Numerical features for imputation and scaling
        numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

        # Imputation
        input_data[numerical_cols] = imputer.transform(input_data[numerical_cols])

        # Scaling
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f"**Loan Approved!** 🎉 (Probability: {prediction_proba[0][1]*100:.2f}%)")
        else:
            st.error(f"**Loan Not Approved.** 🙁 (Probability: {prediction_proba[0][0]*100:.2f}%)")

        st.markdown("---")
        st.info("Disclaimer: This is a demo for educational purposes. Real loan decisions involve complex factors.")

    except ValueError as e:
        st.error(f"Prediction Error: {e}. This usually means an input value was not seen during model training or there's a data type mismatch.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
