"""
TUSK AI — Fraud Detection Model Training Pipeline
==================================================
B.Tech Final Year Project
Dataset: Dataset_TR.csv (10,000 transactions, 15 features)
Models: Logistic Regression (baseline) + Random Forest (main) + XGBoost (advanced)
Output: Trained models saved to ../models/
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "Dataset_TR.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "ml", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

print("=" * 65)
print("  TUSK AI — Fraud Detection Model Training")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – DATA LOADING & EDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 1] Loading Dataset & Basic EDA")

df = pd.read_csv(DATA_PATH)
print(f"  Dataset shape : {df.shape}")
print(f"  Columns       : {list(df.columns)}")
print(f"\n  First 3 rows:\n{df.head(3).to_string()}")

fraud_counts = df['is_fraud'].value_counts()
fraud_pct    = (fraud_counts[1] / len(df)) * 100
print(f"\n  Fraud distribution:")
print(f"    Legitimate : {fraud_counts[0]:,}  ({100-fraud_pct:.1f}%)")
print(f"    Fraud      : {fraud_counts[1]:,}  ({fraud_pct:.1f}%)")
print(f"\n  Missing values:\n{df.isnull().sum()}")

# ── EDA Plots ─────────────────────────────────────────────────────────────────
print("\n  Generating EDA plots …")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("TUSK AI — EDA: Fraud Detection Dataset", fontsize=14, fontweight='bold')

# Fraud ratio
axes[0,0].pie([fraud_counts[0], fraud_counts[1]],
              labels=['Legitimate', 'Fraud'],
              colors=['#10B981', '#EF4444'],
              autopct='%1.1f%%', startangle=90)
axes[0,0].set_title('Fraud vs Legitimate')

# Transaction amount by fraud
df.boxplot(column='transaction_amount', by='is_fraud', ax=axes[0,1])
axes[0,1].set_title('Transaction Amount by Fraud')
axes[0,1].set_xlabel('Is Fraud (0=No, 1=Yes)')

# Account balance
df[df['is_fraud']==0]['account_balance'].hist(ax=axes[0,2], bins=40,
    color='#10B981', alpha=0.7, label='Legit')
df[df['is_fraud']==1]['account_balance'].hist(ax=axes[0,2], bins=40,
    color='#EF4444', alpha=0.7, label='Fraud')
axes[0,2].set_title('Account Balance Distribution')
axes[0,2].legend()

# Amount deviation
df[df['is_fraud']==0]['amount_deviation'].hist(ax=axes[1,0], bins=40,
    color='#10B981', alpha=0.7, label='Legit')
df[df['is_fraud']==1]['amount_deviation'].hist(ax=axes[1,0], bins=40,
    color='#EF4444', alpha=0.7, label='Fraud')
axes[1,0].set_title('Amount Deviation Distribution')
axes[1,0].legend()

# Failed attempts
fraud_fail = df.groupby('is_fraud')['failed_attempts_last_24h'].mean()
axes[1,1].bar(['Legitimate', 'Fraud'], fraud_fail.values,
              color=['#10B981', '#EF4444'])
axes[1,1].set_title('Avg Failed Attempts (last 24h)')

# Foreign transaction fraud rate
df.groupby('is_foreign_transaction')['is_fraud'].mean().plot(
    kind='bar', ax=axes[1,2], color=['#10B981', '#EF4444'], rot=0)
axes[1,2].set_title('Fraud Rate: Foreign vs Domestic')
axes[1,2].set_xticklabels(['Domestic', 'Foreign'])

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'eda_overview.png'), dpi=100, bbox_inches='tight')
plt.close()
print("  ✓ EDA plot saved → ml/plots/eda_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Data Preprocessing")

# Drop ID columns (not useful for ML, leak information)
ID_COLS = ['transaction_id', 'customer_id']
df_clean = df.drop(columns=ID_COLS)
print(f"  Dropped ID columns: {ID_COLS}")

# ── Categorical columns: Label Encoding ───────────────────────────────────────
CAT_COLS = ['transaction_type', 'merchant_category', 'device_type', 'transaction_location']
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    encoders[col] = le
    print(f"  Encoded '{col}' → {list(le.classes_)}")

# Save encoders (needed at inference time)
joblib.dump(encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))
print("\n  ✓ Label encoders saved → models/label_encoders.pkl")

# ── Missing value check ────────────────────────────────────────────────────────
missing = df_clean.isnull().sum().sum()
if missing > 0:
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    print(f"  Filled {missing} missing values with column medians")
else:
    print("  No missing values — clean dataset!")

print(f"\n  Final preprocessed shape: {df_clean.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] Feature Engineering")

# 1. High amount flag: transaction > 2x avg for that user
df_clean['high_amount_flag'] = (
    df_clean['transaction_amount'] > (df_clean['avg_transaction_amount'] * 2)
).astype(int)

# 2. Amount-to-balance ratio: high ratio = risky
df_clean['amount_to_balance_ratio'] = (
    df_clean['transaction_amount'] / (df_clean['account_balance'] + 1e-6)
).round(4)

# 3. Frequency risk score: many transactions in 24h is suspicious
df_clean['frequency_risk'] = df_clean['transaction_frequency_24h'].apply(
    lambda x: 2 if x > 10 else (1 if x > 5 else 0)
)

# 4. Compound risk: multiple risk signals together
df_clean['compound_risk'] = (
    df_clean['is_foreign_transaction']
    + df_clean['high_amount_flag']
    + df_clean['failed_attempts_last_24h'].apply(lambda x: 1 if x >= 3 else 0)
)

print("  ✓ Created 4 new features:")
print("      high_amount_flag        — txn > 2× user's average")
print("      amount_to_balance_ratio — how much of balance is spent")
print("      frequency_risk          — velocity score (0/1/2)")
print("      compound_risk           — sum of risk signals")
print(f"\n  New shape: {df_clean.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Train-Test Split (80 / 20)")

FEATURE_COLS = [c for c in df_clean.columns if c != 'is_fraud']
X = df_clean[FEATURE_COLS]
y = df_clean['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"  Train fraud rate: {y_train.mean():.3f}  |  Test fraud rate: {y_test.mean():.3f}")


# ── Feature scaling (needed for Logistic Regression) ─────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
print("  ✓ Scaler saved → models/scaler.pkl")


# ── SMOTE: Handle class imbalance for training ────────────────────────────────
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
X_train_scaled_res, _ = sm.fit_resample(X_train_scaled, y_train)
print(f"\n  After SMOTE resampling:")
print(f"    Train size: {X_train_res.shape[0]}  "
      f"(fraud: {y_train_res.sum()}, legit: {(y_train_res==0).sum()})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Model Training")

results = {}

# ── 5a. Logistic Regression (Baseline) ───────────────────────────────────────
print("\n  Training Logistic Regression (baseline) …")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled_res, y_train_res)

lr_pred      = lr.predict(X_test_scaled)
lr_prob      = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = {
    'accuracy' : accuracy_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'recall'   : recall_score(y_test, lr_pred),
    'f1'       : f1_score(y_test, lr_pred),
    'roc_auc'  : roc_auc_score(y_test, lr_prob),
}
print(f"  ✓ LR: Acc={results['Logistic Regression']['accuracy']:.3f}  "
      f"Recall={results['Logistic Regression']['recall']:.3f}  "
      f"ROC={results['Logistic Regression']['roc_auc']:.3f}")

# ── 5b. Random Forest (Main Model) ───────────────────────────────────────────
print("\n  Training Random Forest (main model) …")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = {
    'accuracy' : accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall'   : recall_score(y_test, rf_pred),
    'f1'       : f1_score(y_test, rf_pred),
    'roc_auc'  : roc_auc_score(y_test, rf_prob),
}
print(f"  ✓ RF: Acc={results['Random Forest']['accuracy']:.3f}  "
      f"Recall={results['Random Forest']['recall']:.3f}  "
      f"ROC={results['Random Forest']['roc_auc']:.3f}")

# ── 5c. XGBoost ───────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    print("\n  Training XGBoost …")
    scale_pos = (y_train_res==0).sum() / (y_train_res==1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.05, scale_pos_weight=scale_pos,
        random_state=42, n_jobs=-1,
        eval_metric='logloss', verbosity=0
    )
    xgb_model.fit(X_train_res, y_train_res)
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'accuracy' : accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred),
        'recall'   : recall_score(y_test, xgb_pred),
        'f1'       : f1_score(y_test, xgb_pred),
        'roc_auc'  : roc_auc_score(y_test, xgb_prob),
    }
    print(f"  ✓ XGB: Acc={results['XGBoost']['accuracy']:.3f}  "
          f"Recall={results['XGBoost']['recall']:.3f}  "
          f"ROC={results['XGBoost']['roc_auc']:.3f}")
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    print("  ✓ XGBoost saved → models/xgb_model.pkl")
except ImportError:
    print("  [SKIP] XGBoost not installed, skipping.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – RISK SCORING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Risk Scoring Demonstration")

def get_risk_label(prob):
    """Convert probability to human-readable risk tier."""
    if prob < 0.30:  return "LOW",    "#10B981"
    if prob < 0.60:  return "MEDIUM", "#F59E0B"
    if prob < 0.80:  return "HIGH",   "#EF4444"
    return               "CRITICAL", "#7F1D1D"

print("\n  Sample risk scores (first 10 test transactions):")
print(f"  {'#':<4} {'Prob':>8} {'Score%':>8} {'Risk':>10} {'Actual':>8}")
print("  " + "-" * 44)
for i, (prob, actual) in enumerate(zip(rf_prob[:10], y_test.values[:10])):
    score = prob * 100
    label, _ = get_risk_label(prob)
    print(f"  {i+1:<4} {prob:>8.4f} {score:>7.1f}%  {label:>10}  {'FRAUD' if actual else 'LEGIT':>8}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 – MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 7] Full Model Evaluation")

print("\n  ── Comparison Table ──────────────────────────────────────────")
print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'ROC-AUC':>9}")
print("  " + "-" * 68)
for name, m in results.items():
    print(f"  {name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
          f"{m['recall']:>8.4f} {m['f1']:>7.4f} {m['roc_auc']:>9.4f}")

print("\n  ── Random Forest — Classification Report ──────────────────────")
print(classification_report(y_test, rf_pred, target_names=['Legitimate', 'Fraud']))

# ── Confusion Matrix plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("TUSK AI — Model Evaluation", fontsize=13, fontweight='bold')

# Confusion matrices
for ax, (name, pred) in zip(axes[:2], [
    ('Random Forest', rf_pred), ('Logistic Regression', lr_pred)
]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Greens', linewidths=0.5,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    ax.set_title(f'{name}\nConfusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

# ROC curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
axes[2].plot(fpr_rf, tpr_rf, color='#10B981', lw=2,
             label=f"Random Forest (AUC={results['Random Forest']['roc_auc']:.3f})")
axes[2].plot(fpr_lr, tpr_lr, color='#3B82F6', lw=2,
             label=f"Logistic Reg  (AUC={results['Logistic Regression']['roc_auc']:.3f})")
if 'XGBoost' in results:
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
    axes[2].plot(fpr_xgb, tpr_xgb, color='#F59E0B', lw=2,
                 label=f"XGBoost       (AUC={results['XGBoost']['roc_auc']:.3f})")
axes[2].plot([0,1], [0,1], 'k--', lw=1)
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curves')
axes[2].legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'evaluation.png'), dpi=100, bbox_inches='tight')
plt.close()
print("\n  ✓ Evaluation plot saved → ml/plots/evaluation.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 – REAL-TIME SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 8] Real-Time Single Transaction Prediction (Simulation)")

def predict_transaction(transaction_dict, rf_model, scaler_obj, encoders_obj, feature_cols):
    """
    Simulate real-time fraud detection for a single transaction.
    Returns: (is_fraud: bool, risk_score: float, risk_label: str)
    """
    df_tx = pd.DataFrame([transaction_dict])

    # Drop ID columns if present
    df_tx.drop(columns=['transaction_id', 'customer_id'], errors='ignore', inplace=True)

    # Encode categoricals
    for col, le in encoders_obj.items():
        if col in df_tx.columns:
            val = df_tx[col].astype(str).values[0]
            if val not in le.classes_:
                val = le.classes_[0]   # fallback to first known class
            df_tx[col] = le.transform([val])

    # Feature engineering (same as training)
    df_tx['high_amount_flag'] = int(
        df_tx['transaction_amount'].values[0] > df_tx['avg_transaction_amount'].values[0] * 2)
    df_tx['amount_to_balance_ratio'] = round(
        df_tx['transaction_amount'].values[0] / (df_tx['account_balance'].values[0] + 1e-6), 4)
    freq = df_tx['transaction_frequency_24h'].values[0]
    df_tx['frequency_risk'] = 2 if freq > 10 else (1 if freq > 5 else 0)
    df_tx['compound_risk'] = (
        df_tx['is_foreign_transaction'].values[0]
        + df_tx['high_amount_flag'].values[0]
        + (1 if df_tx['failed_attempts_last_24h'].values[0] >= 3 else 0)
    )

    # Align columns
    df_tx = df_tx.reindex(columns=feature_cols, fill_value=0)

    prob  = rf_model.predict_proba(df_tx)[0][1]
    label, color = get_risk_label(prob)
    return bool(prob >= 0.5), round(float(prob) * 100, 2), label

# Load objects for demo
_enc_demo = encoders
_scaler_demo = scaler

# Sample suspicious transaction
test_tx = {
    'age': 28, 'account_balance': 1200.00,
    'transaction_amount': 4500.00,        # way above balance
    'transaction_type': df['transaction_type'].iloc[0],
    'merchant_category': df['merchant_category'].iloc[0],
    'device_type': df['device_type'].iloc[0],
    'transaction_location': df['transaction_location'].iloc[0],
    'is_foreign_transaction': 1,           # foreign
    'failed_attempts_last_24h': 4,         # multiple failures
    'avg_transaction_amount': 250.00,
    'transaction_frequency_24h': 12,       # high velocity
    'amount_deviation': 18.0
}

is_fraud, risk_score, risk_label = predict_transaction(
    test_tx, rf, _scaler_demo, _enc_demo, FEATURE_COLS
)
print(f"\n  Test Transaction:")
print(f"    Amount       : ${test_tx['transaction_amount']:,.2f}")
print(f"    Balance      : ${test_tx['account_balance']:,.2f}")
print(f"    Foreign      : {'Yes' if test_tx['is_foreign_transaction'] else 'No'}")
print(f"    Failed Att.  : {test_tx['failed_attempts_last_24h']}")
print(f"  → Risk Score   : {risk_score}%")
print(f"  → Risk Label   : {risk_label}")
print(f"  → Prediction   : {'⚠️  FRAUD DETECTED' if is_fraud else '✅ LEGITIMATE'}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 – FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 9] Feature Importance (Random Forest)")

importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
importances_sorted = importances.sort_values(ascending=False)

print("\n  Top 10 Most Important Features:")
for i, (feat, imp) in enumerate(importances_sorted.head(10).items(), 1):
    bar = '█' * int(imp * 200)
    print(f"  {i:>2}. {feat:<35} {imp:.4f}  {bar}")

# Save feature importance plot
plt.figure(figsize=(10, 6))
importances_sorted.head(12).plot(kind='barh', color='#10B981', edgecolor='white')
plt.title('TUSK AI — Feature Importance (Random Forest)', fontweight='bold')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'feature_importance.png'), dpi=100, bbox_inches='tight')
plt.close()
print("\n  ✓ Feature importance plot saved → ml/plots/feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 – SAVE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 10] Saving Models")

joblib.dump(rf,   os.path.join(MODEL_DIR, 'rf_model.pkl'))
joblib.dump(lr,   os.path.join(MODEL_DIR, 'lr_model.pkl'))
joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, 'feature_cols.pkl'))

print("  ✓ Random Forest  → models/rf_model.pkl")
print("  ✓ Logistic Reg   → models/lr_model.pkl")
print("  ✓ Scaler         → models/scaler.pkl")
print("  ✓ Encoders       → models/label_encoders.pkl")
print("  ✓ Feature cols   → models/feature_cols.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ✅ Model Training Complete!")
print(f"  Best model: Random Forest")
print(f"    Accuracy  : {results['Random Forest']['accuracy']:.4f}")
print(f"    Recall    : {results['Random Forest']['recall']:.4f}")
print(f"    Precision : {results['Random Forest']['precision']:.4f}")
print(f"    ROC-AUC   : {results['Random Forest']['roc_auc']:.4f}")
print("\n  Models saved in: ./models/")
print("  Plots  saved in: ./ml/plots/")
print("\n  ➜ Next: run  python backend/main.py  to start the API server")
print("=" * 65)
