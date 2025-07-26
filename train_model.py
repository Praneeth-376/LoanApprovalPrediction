import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# --- 1. Load your existing datasets ---
try:
    df_train = pd.read_csv('/train.csv')
    df_test = pd.read_csv('/test.csv') # Load test data
    print("Train and Test datasets loaded successfully.")
except FileNotFoundError:
    print("Error: train.csv or test.csv not found. Please ensure they are in the same directory.")
    exit() # Exit if files are not found

# Identify target variable and features from the training data
# Assuming 'Loan_Status' is your target variable in train.csv
X_train_df = df_train.drop('Loan_Status', axis=1)
y_train = df_train['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0) # Convert Y/N to 1/0

# X_test_df already contains only features, as 'Loan_Status' is not in test.csv
X_test_df = df_test.copy() # Make a copy to avoid SettingWithCopyWarning later if needed

# Identify categorical and numerical features from your dataset
# Adjust these lists based on the actual columns in your train.csv and test.csv
categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# --- 2. Preprocessing and Model Training ---

# 2.1. Label Encoding for Categorical Features
# Fit LabelEncoders on the training data for consistency
le_gender = LabelEncoder()
le_married = LabelEncoder()
le_dependents = LabelEncoder()
le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_property_area = LabelEncoder()

# Fit and transform training data
X_train_df['Gender'] = le_gender.fit_transform(X_train_df['Gender'])
X_train_df['Married'] = le_married.fit_transform(X_train_df['Married'])
X_train_df['Dependents'] = le_dependents.fit_transform(X_train_df['Dependents'])
X_train_df['Education'] = le_education.fit_transform(X_train_df['Education'])
X_train_df['Self_Employed'] = le_self_employed.fit_transform(X_train_df['Self_Employed'])
X_train_df['Property_Area'] = le_property_area.fit_transform(X_train_df['Property_Area'])

# Transform test data using the *fitted* encoders (important to use .transform, not .fit_transform)
for col, encoder in zip(categorical_features, [le_gender, le_married, le_dependents, le_education, le_self_employed, le_property_area]):
    # Note: If there are new categories in test.csv not seen in train.csv, this will raise an error.
    # A robust solution would involve OneHotEncoder or handling unknown categories.
    # For now, we assume test categories are a subset of train categories.
    X_test_df[col] = encoder.transform(X_test_df[col])


# 2.2. Impute missing numerical values (if any)
imputer = SimpleImputer(strategy='mean')
# Fit imputer ONLY on training data
X_train_df[numerical_features] = imputer.fit_transform(X_train_df[numerical_features])
# Transform test data using the *fitted* imputer
X_test_df[numerical_features] = imputer.transform(X_test_df[numerical_features])


# 2.3. Feature Scaling for Numerical Features
scaler = StandardScaler()
# Fit scaler ONLY on training data
X_train_df[numerical_features] = scaler.fit_transform(X_train_df[numerical_features])
# Transform test data using the *fitted* scaler
X_test_df[numerical_features] = scaler.transform(X_test_df[numerical_features])


# 2.4. Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_df, y_train)

# --- 3. Save the fitted components ---
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_married, 'le_married.pkl')
joblib.dump(le_dependents, 'le_dependents.pkl')
joblib.dump(le_education, 'le_education.pkl')
joblib.dump(le_self_employed, 'le_self_employed.pkl')
joblib.dump(le_property_area, 'le_property_area.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')

print("All preprocessors and model saved successfully!")
print("Run the next cell to start the Streamlit app.")
