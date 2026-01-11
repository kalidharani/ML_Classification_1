import streamlit as st
import numpy as np
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Logistic Regression Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Logistic Regression Prediction App")
st.write("This app uses a trained Logistic Regression model to make predictions.")

# -------------------------------
# Load PKL File (SAFE LOADING)
# -------------------------------
loaded_object = joblib.load("logistic_regression_model.pkl")

# If PKL contains a dictionary, extract the model
if isinstance(loaded_object, dict):
    model = loaded_object["model"]
else:
    model = loaded_object

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Input Values")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
salary = st.number_input("Salary", min_value=1000, max_value=1000000, value=30000)

# -------------------------------
# Encode Input (same as training)
# Male = 1, Female = 0
# -------------------------------
gender_encoded = 1 if gender == "Male" else 0

input_data = np.array([[gender_encoded, age, salary]])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Prediction: YES")
    else:
        st.error("❌ Prediction: NO")
