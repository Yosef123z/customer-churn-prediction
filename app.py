"""
Customer Churn Prediction App
=============================
A Streamlit web application for predicting customer churn using a trained
Random Forest classifier.

Feature Order: Age, Gender, Tenure, MonthlyCharges, InternetService, TechSupport

To run the app: streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# App header
st.title("📉 Customer Churn Prediction")
st.markdown("---")
st.write("Enter customer details below and click **Predict** to see if they are likely to churn.")
st.markdown("---")

# --- INPUT FIELDS ---
col1, col2 = st.columns(2)

with col1:
    # 1. Age
    age = st.number_input("Age", min_value=10, max_value=100, value=30, help="Customer's age")
    
    # 3. Tenure
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=130, value=10, 
                             help="Number of months the customer has been with the company")
    
    # 5. Internet Service (None=0, DSL=1, Fiber optic=2)
    internet = st.selectbox("Internet Service", ["None", "DSL", "Fiber optic"])

with col2:
    # 2. Gender (Male=0, Female=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    # 4. Monthly Charges
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, 
                                       value=50.0, help="Customer's monthly billing amount")
    
    # 6. Tech Support (No=0, Yes=1)
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])

st.markdown("---")

# Prediction button
predict_button = st.button("🔮 Predict", use_container_width=True)

if predict_button:
    # --- MAPPING INPUTS TO NUMBERS ---
    gender_map = {"Male": 0, "Female": 1}
    internet_map = {"None": 0, "DSL": 1, "Fiber optic": 2}
    tech_support_map = {"No": 0, "Yes": 1}

    # Map the user inputs
    gender_val = gender_map[gender]
    internet_val = internet_map[internet]
    tech_support_val = tech_support_map[tech_support]

    # --- PREDICTION ---
    # Create feature list in exact order:
    # ['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'InternetService', 'TechSupport']
    X_new = [age, gender_val, tenure, monthly_charges, internet_val, tech_support_val]

    # Convert to numpy array
    X_array = np.array(X_new).reshape(1, -1)

    # Scale the features
    X_scaled = scaler.transform(X_array)

    # Predict
    prediction = model.predict(X_scaled)[0]

    # Output with visual feedback
    st.markdown("### Prediction Result")
    
    if prediction == 1:
        st.error("⚠️ **Prediction: Will Churn**")
        st.markdown("This customer is at **high risk** of leaving. Consider implementing retention strategies.")
    else:
        st.balloons()
        st.success("✅ **Prediction: Will Not Churn**")
        st.markdown("This customer appears to be **satisfied** and is likely to stay.")

else:
    st.info("👆 Click **Predict** to see the result.")
