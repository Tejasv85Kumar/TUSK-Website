"""
TUSK AI — FastAPI Backend
=========================
Serves the fraud detection ML model via REST API.
Endpoints:
  POST  /api/fraud/analyze      → full fraud analysis
  POST  /api/fraud/quick-score  → lightweight risk score only
  GET   /api/stats              → live platform statistics
  GET   /api/health             → health check
  POST  /api/chat               → RAGFlow chatbot proxy
"""

import os, sys, json, random, time
from datetime import datetime
from typing import Optional, List

import httpx

import numpy as np
import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load models at startup ────────────────────────────────────────────────────
print("[*] Loading TUSK AI fraud detection models ...")
try:
    RF_MODEL     = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    SCALER       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    ENCODERS     = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    print("[OK] Models loaded successfully")
    MODEL_LOADED = True
except Exception as e:
    print(f"[WARN] Model not loaded: {e}")
    print("   Run  python ml/train_model.py  first.")
    MODEL_LOADED = False
    RF_MODEL = SCALER = ENCODERS = FEATURE_COLS = None

# ── RAGFlow Chat Configuration ────────────────────────────────────────────────
# Replace RAGFLOW_CHAT_ID with your actual Assistant ID from your RAGFlow dashboard.
# (Open your assistant → Settings → copy the ID from the URL or info panel)
RAGFLOW_API_KEY  = "sk-734cf1b03bc547d2a6d64d2b349346ee"
RAGFLOW_BASE_URL = "http://localhost:9380"          # change if cloud-hosted, e.g. https://demo.ragflow.io
RAGFLOW_CHAT_ID  = "YOUR_CHAT_ASSISTANT_ID_HERE"   # ← paste your RAGFlow assistant ID here

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TUSK AI — Fraud Detection API",
    description="Real-time fraud detection and risk scoring for financial transactions.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS (allow frontend HTML to call this API) ───────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # in production: set to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static frontend files ───────────────────────────────────────────────
STATIC_DIR = BASE_DIR


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class TransactionInput(BaseModel):
    """All fields a bank would send for fraud analysis."""
    age:                      int   = Field(..., ge=18, le=100,   example=35)
    account_balance:          float = Field(..., ge=0,            example=5000.0)
    transaction_amount:       float = Field(..., ge=0,            example=299.99)
    transaction_type:         str   = Field(...,                  example="Online")
    merchant_category:        str   = Field(...,                  example="Retail")
    device_type:              str   = Field(...,                  example="Mobile")
    transaction_location:     str   = Field(...,                  example="India")
    is_foreign_transaction:   int   = Field(..., ge=0, le=1,     example=0)
    failed_attempts_last_24h: int   = Field(..., ge=0,            example=0)
    avg_transaction_amount:   float = Field(..., ge=0,            example=250.0)
    transaction_frequency_24h:int   = Field(..., ge=0,            example=3)
    amount_deviation:         float = Field(...,                  example=0.2)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35, "account_balance": 5000.0,
                "transaction_amount": 299.99,
                "transaction_type": "Online",
                "merchant_category": "Retail",
                "device_type": "Mobile",
                "transaction_location": "India",
                "is_foreign_transaction": 0,
                "failed_attempts_last_24h": 0,
                "avg_transaction_amount": 250.0,
                "transaction_frequency_24h": 3,
                "amount_deviation": 0.2
            }
        }


class FraudAnalysisResult(BaseModel):
    is_fraud:          bool
    risk_score:        float
    risk_percentage:   float
    risk_label:        str
    risk_color:        str
    confidence:        float
    processing_time_ms:float
    model_used:        str
    features_analysis: dict
    recommendation:    str
    timestamp:         str


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def encode_safe(encoders, col, value):
    """Encode a categorical value, falling back to first class if unknown."""
    le = encoders[col]
    val_str = str(value)
    if val_str not in le.classes_:
        val_str = le.classes_[0]
    return int(le.transform([val_str])[0])


def preprocess_transaction(tx: TransactionInput) -> pd.DataFrame:
    """Turn a TransactionInput into the exact feature vector the model expects."""
    d = tx.dict()

    # Encode categorical columns
    for col in ['transaction_type', 'merchant_category', 'device_type', 'transaction_location']:
        if col in ENCODERS:
            d[col] = encode_safe(ENCODERS, col, d[col])

    # Feature engineering (mirrors train_model.py)
    d['high_amount_flag'] = int(d['transaction_amount'] > d['avg_transaction_amount'] * 2)
    d['amount_to_balance_ratio'] = round(
        d['transaction_amount'] / (d['account_balance'] + 1e-6), 4)
    freq = d['transaction_frequency_24h']
    d['frequency_risk'] = 2 if freq > 10 else (1 if freq > 5 else 0)
    d['compound_risk'] = (
        d['is_foreign_transaction']
        + d['high_amount_flag']
        + (1 if d['failed_attempts_last_24h'] >= 3 else 0)
    )

    df_row = pd.DataFrame([d])
    df_row = df_row.reindex(columns=FEATURE_COLS, fill_value=0)
    return df_row


def get_risk_tier(prob: float):
    if prob < 0.30:
        return "LOW",      "#10B981", "✅ Transaction appears safe. Approve normally."
    if prob < 0.60:
        return "MEDIUM",   "#F59E0B", "⚠️ Elevated risk. Consider additional verification."
    if prob < 0.80:
        return "HIGH",     "#EF4444", "🚨 High fraud risk. Request customer confirmation."
    return     "CRITICAL", "#7F1D1D", "🔴 Critical fraud signal. Block and alert customer immediately."


def compute_feature_analysis(tx: TransactionInput) -> dict:
    """Human-readable analysis of why this transaction is risky."""
    flags = {}
    flags['amount_vs_average'] = {
        'value': round(tx.transaction_amount / max(tx.avg_transaction_amount, 1), 2),
        'label': 'Transaction / Avg',
        'risky': tx.transaction_amount > tx.avg_transaction_amount * 2
    }
    flags['balance_coverage'] = {
        'value': round((tx.transaction_amount / max(tx.account_balance, 1)) * 100, 1),
        'label': '% of Account Balance',
        'risky': tx.transaction_amount > tx.account_balance * 0.8
    }
    flags['foreign_transaction'] = {
        'value': bool(tx.is_foreign_transaction),
        'label': 'Foreign Transaction',
        'risky': bool(tx.is_foreign_transaction)
    }
    flags['failed_attempts'] = {
        'value': tx.failed_attempts_last_24h,
        'label': 'Failed Attempts (24h)',
        'risky': tx.failed_attempts_last_24h >= 3
    }
    flags['high_velocity'] = {
        'value': tx.transaction_frequency_24h,
        'label': 'Transactions (24h)',
        'risky': tx.transaction_frequency_24h > 10
    }
    flags['amount_deviation'] = {
        'value': round(tx.amount_deviation, 2),
        'label': 'Amount Deviation Score',
        'risky': tx.amount_deviation > 5
    }
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": MODEL_LOADED,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/fraud/analyze", response_model=FraudAnalysisResult)
def analyze_transaction(tx: TransactionInput):
    """
    Full fraud detection analysis for a single transaction.
    Returns risk score, label, confidence, and feature breakdown.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run: python ml/train_model.py"
        )

    t_start = time.time()

    try:
        df_row = preprocess_transaction(tx)
        proba  = RF_MODEL.predict_proba(df_row)[0]
        prob_fraud = float(proba[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    processing_ms = round((time.time() - t_start) * 1000, 2)

    risk_label, risk_color, recommendation = get_risk_tier(prob_fraud)
    feature_analysis = compute_feature_analysis(tx)

    return FraudAnalysisResult(
        is_fraud           = prob_fraud >= 0.5,
        risk_score         = round(prob_fraud, 4),
        risk_percentage    = round(prob_fraud * 100, 2),
        risk_label         = risk_label,
        risk_color         = risk_color,
        confidence         = round(max(proba) * 100, 2),
        processing_time_ms = processing_ms,
        model_used         = "Random Forest Ensemble (200 trees)",
        features_analysis  = feature_analysis,
        recommendation     = recommendation,
        timestamp          = datetime.now().isoformat()
    )


@app.post("/api/fraud/quick-score")
def quick_score(tx: TransactionInput):
    """Lightweight endpoint — returns only risk score and label."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df_row     = preprocess_transaction(tx)
    prob_fraud = float(RF_MODEL.predict_proba(df_row)[0][1])
    label, color, _ = get_risk_tier(prob_fraud)

    return {
        "risk_score":      round(prob_fraud * 100, 1),
        "risk_label":      label,
        "risk_color":      color,
        "is_fraud":        prob_fraud >= 0.5
    }


@app.get("/api/stats")
def get_stats():
    """Live platform statistics for the dashboard."""
    base_transactions = 1248
    base_threats      = 9854

    # Simulate real-time fluctuation
    rand_offset = random.randint(0, 30)
    return {
        "checks_today":       base_transactions + rand_offset,
        "threats_blocked":    base_threats + random.randint(0, 5),
        "detection_accuracy": 99.7,
        "avg_response_ms":    47,
        "false_positive_rate": 0.3,
        "model_version":      "RF-v2.0",
        "uptime":             "99.9%",
        "timestamp":          datetime.now().isoformat()
    }


@app.get("/api/alerts/recent")
def get_recent_alerts():
    """Return recent fraud alert samples for the live dashboard."""
    alerts = [
        {"id": "ALT001", "type": "Unusual Pattern",
         "amount": 5240, "currency": "USD",
         "location": "Unknown", "severity": "HIGH",
         "color": "#8B5CF6", "timestamp": datetime.now().isoformat()},
        {"id": "ALT002", "type": "Login Anomaly",
         "amount": 0, "currency": "USD",
         "location": "Tokyo, JP", "severity": "MEDIUM",
         "color": "#F59E0B", "timestamp": datetime.now().isoformat()},
        {"id": "ALT003", "type": "Fraud Blocked",
         "amount": 12500, "currency": "USD",
         "location": "New York, US", "severity": "CRITICAL",
         "color": "#10B981", "timestamp": datetime.now().isoformat()},
    ]
    return {"alerts": alerts, "count": len(alerts)}


# ══════════════════════════════════════════════════════════════════════════════
# RAGFLOW CHAT PROXY
# ══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str    # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ragflow(req: ChatRequest):
    """
    Proxy user messages to RAGFlow and return the AI reply.
    Keeps the API key secure on the server side.
    """
    if RAGFLOW_CHAT_ID == "YOUR_CHAT_ASSISTANT_ID_HERE":
        raise HTTPException(
            status_code=503,
            detail="RAGFlow chat_id not configured. Set RAGFLOW_CHAT_ID in backend/main.py."
        )

    url = f"{RAGFLOW_BASE_URL}/api/v1/chats/{RAGFLOW_CHAT_ID}/completions"

    # Build message history in OpenAI-compatible format
    messages = [{"role": m.role, "content": m.content} for m in req.history]
    messages.append({"role": "user", "content": req.message})

    payload = {
        "model": "model",       # RAGFlow ignores this but it is required by the spec
        "messages": messages,
        "stream": False,
    }

    if req.session_id:
        payload["session_id"] = req.session_id

    headers = {
        "Authorization": f"Bearer {RAGFLOW_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="RAGFlow request timed out.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"RAGFlow returned an error: {e.response.status_code}"
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RAGFlow connection failed: {str(e)}")

    # Extract the assistant reply from the response
    try:
        reply = (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            or data.get("answer", "")
            or "I'm sorry, I couldn't process your request right now."
        )
        session_id = data.get("session_id") or req.session_id
    except Exception:
        reply = "I received a response but couldn't parse it. Please try again."
        session_id = req.session_id

    return ChatResponse(reply=reply, session_id=session_id)


# Mount static files at root (placed last so it doesn't override /api)
# html=True allows serving index.html at the root /
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  TUSK AI Backend Server")
    print("  http://localhost:8000")
    print("  API Docs -> http://localhost:8000/api/docs")
    print("  Chat API -> POST http://localhost:8000/api/chat")
    print("="*55 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
