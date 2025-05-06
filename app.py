# app.py

import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Train the model (only once)
MODEL_PATH = "model.pkl"

def train_and_save_model():
    if not os.path.exists(MODEL_PATH):
        # Use 3 numeric features: amount, location_code, time_of_day
        X, y = make_classification(n_samples=1000, n_features=3,
                                   n_informative=3, n_redundant=0,
                                   random_state=42)
        model = RandomForestClassifier()
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Map city to numeric code
city_to_code = {
    "Mumbai": 0,
    "Delhi": 1,
    "Bangalore": 2,
    "Chennai": 3,
    "Kolkata": 4
}

# Predict function
def predict_fraud(model, features):
    features_array = np.array([features]).reshape(1, -1)
    prediction = model.predict(features_array)
    return bool(prediction[0])

# Main App
def main():
    st.set_page_config(page_title="AI Credit Card Fraud Detector")
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("Enter transaction details below to check if it's fraudulent.")

    # UI Inputs
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=1.0)
    location = st.selectbox("Transaction Location", list(city_to_code.keys()))
    time_of_day = st.slider("Time of Day (0 = midnight, 23 = 11PM)", 0, 23)

    if st.button("Check for Fraud"):
        train_and_save_model()
        model = load_model()
        location_code = city_to_code[location]
        is_fraud = predict_fraud(model, [amount, location_code, time_of_day])

        if is_fraud:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

if __name__ == "__main__":
    main()
