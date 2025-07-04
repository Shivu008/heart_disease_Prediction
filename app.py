import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page Title
st.title("💓 Heart Disease Prediction App")
st.write("Fill the following details to check the risk of heart disease:")

# Input Fields
age = st.slider("Age", 29, 77, 54)
gender = st.selectbox("Gender", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
rest_bp = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
chol = st.slider("Cholesterol (mg/dL)", 150, 400, 230)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
max_hr = st.slider("Maximum Heart Rate", 70, 202, 140)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0)
slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Encoding
gender = 1 if gender == "Male" else 0
fasting_bs = 1 if fasting_bs == "Yes" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0

# One-hot encoding for categorical features
data = {
    'Age': age,
    'Resting_BP': rest_bp,
    'Cholesterol': chol,
    'Fasting_Blood_Sugar': fasting_bs,
    'Max_Heart_Rate': max_hr,
    'Oldpeak': oldpeak,
    'Num_Major_Vessels': vessels,
    'Gender': gender,
    'Chest_Pain_Type_Atypical Angina': 1 if chest_pain == "Atypical Angina" else 0,
    'Chest_Pain_Type_Non-Anginal Pain': 1 if chest_pain == "Non-Anginal Pain" else 0,
    'Chest_Pain_Type_Asymptomatic': 1 if chest_pain == "Asymptomatic" else 0,
    'Resting_ECG_Left Ventricular Hypertrophy': 1 if rest_ecg == "Left Ventricular Hypertrophy" else 0,
    'Resting_ECG_ST-T Abnormality': 1 if rest_ecg == "ST-T Abnormality" else 0,
    'Exercise_Induced_Angina': exercise_angina,
    'Slope_Flat': 1 if slope == "Flat" else 0,
    'Slope_Downsloping': 1 if slope == "Downsloping" else 0,
    'Thalassemia_Fixed Defect': 1 if thal == "Fixed Defect" else 0,
    'Thalassemia_Reversible Defect': 1 if thal == "Reversible Defect" else 0
}

# Create DataFrame
input_df = pd.DataFrame([data])

# Scale the input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ Risk of Heart Disease Detected!")
    else:
        st.success("✅ No Risk of Heart Disease Detected.")
