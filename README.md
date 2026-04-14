#  TUSK: AI-Powered Fraud Detection System

## Website Link

https://tusk-jade.vercel.app/

##  Project Overview

TUSK is an AI-powered fraud detection system designed to identify suspicious banking transactions in real-time. The system uses machine learning techniques to analyze transaction patterns and assign a **risk score** to each transaction, helping prevent financial fraud efficiently.

---

##  Objectives

* Detect fraudulent transactions using machine learning
* Assign a **risk score (0 to 1)** for each transaction
* Simulate **real-time fraud detection**
* Reduce false positives and improve detection accuracy

---

##  Key Features

*  Fraud Detection using ML models
*  Risk Scoring System (Low, Medium, High)
*  Real-time transaction prediction (simulation)
*  Feature importance analysis
*  Model saving & reuse

---

##  Dataset

The dataset is **synthetically generated** (due to privacy constraints of banking data).

### Features Used:

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
* is_fraud (Target Variable)

---

##  Tech Stack

* Python 
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebook / VS Code

---

##  Workflow

### 1. Data Preprocessing

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Dropping unnecessary columns

### 2. Model Training

* Logistic Regression (Baseline)
* Random Forest (Main Model)

### 3. Model Evaluation

* Accuracy
* Precision
* Recall
* Confusion Matrix

### 4. Risk Scoring System

Instead of only predicting fraud (0/1), the model outputs a probability:

| Score     | Risk Level  |
| --------- | ----------- |
| 0.0 – 0.3 | Low Risk    |
| 0.3 – 0.7 | Medium Risk |
| 0.7 – 1.0 | High Risk   |

---

## 🏦 Real-World Use Case

* Low Risk → Allow transaction
* Medium Risk → Request OTP verification
* High Risk → Block transaction & alert user

---

##  Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/tusk-fraud-detection.git

# Navigate to project folder
cd tusk-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

---

##  How to Run

```bash
python main.py
```

Or run the Jupyter Notebook:

```bash
jupyter notebook
```

---

##  Example Output

```python
Transaction Risk Score: 0.87
Risk Level: High Risk 
```

---

##  Model Saving

The trained model is saved using:

```python
import pickle
pickle.dump(model, open("fraud_model.pkl", "wb"))
```

---

##  Limitations

* Uses synthetic dataset (not real banking data)
* Real-time system is simulated
* No live API integration (yet)

---

##  Future Scope

* Real-time API integration with banking systems
* Advanced models (XGBoost, LSTM)
* Dashboard for fraud monitoring
* Integration with chatbot for user interaction

---

##  Team Members

* Tejasv Kumar
* Utkarsh Sachan
* Utkarsh Tiwari
* Shikhar Shukla
* Ayush Sachan

---

## 📜 License

This project is for academic purposes only.
