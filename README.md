#  TUSK: AI-Powered Fraud Detection and Banking Assistant

## Website Link

https://tusk-jade.vercel.app/

## Project Overview

**TUSK** is an AI-powered system designed to enhance banking security and customer experience by combining **real-time fraud detection** with an **intelligent banking assistant**.

The system analyzes transaction data using machine learning to detect fraudulent activities instantly and assigns a **risk score** to each transaction. In parallel, it provides a **24/7 AI-powered assistant** to help users with banking queries, fraud reporting, and account-related actions.

---

## Objectives

* Detect fraudulent transactions in real-time
* Assign dynamic **risk scores** to transactions
* Provide instant fraud alerts and prevention mechanisms
* Offer automated banking support using AI
* Reduce response time and improve customer experience

---

## Key Features

### Fraud Detection System

* Machine Learning-based fraud detection
* Real-time transaction monitoring (simulation)
* Risk scoring system (Low, Medium, High)
* Feature-based anomaly detection
* Reduced false positives using behavioral analysis

### Banking Assistant

* AI-powered chatbot for banking queries
* Handles:

  * Balance inquiries
  * Transaction history
  * Fraud reporting
  * Account security actions
* 24/7 automated customer support

### Security & Risk Management

* Instant fraud alerts
* High-risk transaction blocking
* OTP / verification for medium-risk cases
* Secure API-based integration (conceptual)

---

## Dataset

Due to privacy constraints, the system uses a **synthetic dataset** that simulates real banking transactions.

### Features:

* transaction_id
* customer_id
* age
* account_balance
* transaction_amount
* transaction_type
* merchant_category
* device_type
* transaction_location
* is_foreign_transaction
* failed_attempts_last_24h
* avg_transaction_amount
* transaction_frequency_24h
* amount_deviation
* is_fraud

---

## Tech Stack

### Backend

* Python
* Scikit-learn (ML models)
* TensorFlow (optional for advanced models)
* FastAPI (for API simulation)

### Frontend

* React.js
* Tailwind CSS

### Chatbot

* Rasa / Dialogflow

### Database

* MongoDB / MySQL

### Deployment (Optional)

* AWS / GCP / Azure
* Docker

---

## System Workflow

### 1. Data Processing

* Data cleaning and preprocessing
* Feature engineering
* Encoding and scaling

### 2. Fraud Detection Model

* Logistic Regression (baseline)
* Random Forest (main model)
* Optional: XGBoost / LSTM

### 3. Risk Scoring System

Instead of only predicting fraud (0/1), the model generates a probability:

| Score     | Risk Level  |
| --------- | ----------- |
| 0.0 – 0.3 | Low Risk    |
| 0.3 – 0.7 | Medium Risk |
| 0.7 – 1.0 | High Risk   |

---

### 4. Decision Engine

* Low Risk → Allow transaction
* Medium Risk → Request verification (OTP)
* High Risk → Block transaction & alert user

---

### 5. Banking Assistant Flow

* User interacts with chatbot
* Queries processed using NLP
* Backend fetches required data
* Response generated instantly

---

## Model Evaluation

* Accuracy
* Precision
* Recall
* Confusion Matrix

Focus:

* High **Recall** → Catch fraud
* Balanced **Precision** → Avoid false alarms

---

## Installation & Setup

```bash id="b6u0mx"
# Clone the repository
git clone https://github.com/your-username/tusk-project.git

# Navigate to project directory
cd tusk-project

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Run Backend

```bash id="3wnv66"
python main.py
```

### Run Frontend

```bash id="y1j2p7"
npm install
npm start
```

### Run Chatbot (Rasa)

```bash id="7rj8t2"
rasa run
```

---

## Example Output

```python id="7a0j4k"
Transaction Risk Score: 0.82
Risk Level: High Risk 🚨Action Taken: Transaction Blocked
```

---

## Model Saving

```python id="e7z7bb"
import pickle
pickle.dump(model, open("fraud_model.pkl", "wb"))
```

---

## Limitations

* Uses synthetic dataset
* Real-time system is simulated
* Limited chatbot training data

---

## Future Scope

* Integration with real banking APIs
* Advanced deep learning models (LSTM)
* Live fraud monitoring dashboard
* Voice-enabled banking assistant
* Continuous learning system

---

## Team Members

* Tejasv Kumar
* Utkarsh Sachan
* Utkarsh Tiwari
* Shikhar Shukla
* Ayush Sachan

---
