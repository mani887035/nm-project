# 🛡️ Guarding Transactions with AI-Powered Credit Card Fraud Detection and Prevention

This project is a Streamlit-based web application that detects fraudulent credit card transactions using a machine learning model. It uses synthetic data to simulate real-time fraud prediction and demonstrate the power of AI in financial security.

# live demo app

click here:https: //nm-project-nx6xhvbdz9ldbecihjarez.streamlit.app/

---

## 🚀 Features

- 💡 Trains a Random Forest model on synthetic transaction data
- 🧠 Predicts fraud based on:
  - Transaction amount
  - Transaction location (city)
  - Time of day
- ⚠️ Alerts user if the transaction is suspicious
- 🌐 Interactive UI built with Streamlit

---

## 📁 Project Structure

nm-project/
├── app.py               # Streamlit application for fraud detection
├── creditcard.csv       # Dataset used for model training/testing
├── requirements.txt     # List of Python dependencies
└── README.md            # Project documentation


---

## 🔧 How It Works

1. When you first run the app, it trains a Random Forest Classifier using sklearn’s make_classification().
2. The model is saved as model.pkl.
3. The user inputs transaction details via a web UI.
4. The model predicts whether the transaction is fraudulent or legitimate.

---

## 💻 Tech Stack

- Python 3.8+
- Streamlit
- scikit-learn
- NumPy
- joblib

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/fraud_detection_app.git
cd fraud_detection_app
pip install -r requirements.txt
