import sqlite3
from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, conlist

app = FastAPI(title="Real-Time SOC Dashboard API")

# Database setup
DB_PATH = "detections.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prediction TEXT,
            confidence REAL,
            features TEXT
        )
    ''')
    conn.commit()
    conn.close()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    features: conlist(float, min_length=25, max_length=25)

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    timestamp: str

MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None
class_labels: Optional[List[str]] = None

@app.on_event("startup")
def startup_event():
    init_db()
    load_artifacts()

def load_artifacts():
    global model, scaler, class_labels

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        print(f"Warning: Failed to load model '{MODEL_PATH}': {exc}")

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as exc:
        print(f"Warning: Failed to load scaler '{SCALER_PATH}': {exc}")

    if model is not None:
        if hasattr(model, "classes_"):
            class_labels = [str(label) for label in model.classes_]
        elif hasattr(model, "get_booster"):
            class_labels = ["BENIGN", "ATTACK"]
        else:
            class_labels = ["BENIGN", "ATTACK"]


# Mapping numeric labels to human-readable names (Standard IDS Labels)
LABEL_MAP = {
    "0": "BENIGN",
    "1": "Botnet",
    "2": "DDoS",
    "3": "DoS GoldenEye",
    "4": "DoS Hulk",
    "5": "DoS Slowhttptest",
    "6": "DoS Slowloris",
    "7": "FTP Brute Force",
    "8": "Heartbleed",
    "9": "Infiltration",
    "10": "Port Scan",
    "11": "SSH Brute Force",
    "12": "Web Attack - Brute Force",
    "13": "Web Attack - SQL Injection",
    "14": "Web Attack - XSS"
}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # If model is not loaded, we use a mock prediction for development/testing
    if model is None or scaler is None:
        # Mock logic if files are missing (picking a random threat for demo)
        if np.random.rand() > 0.8:
            label = str(np.random.randint(1, 15))
        else:
            label = "0" # BENIGN
        confidence = round(np.random.uniform(0.85, 0.99), 4)
    else:
        features = np.array(request.features, dtype=float).reshape(1, -1)
        try:
            scaled_features = scaler.transform(features)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to scale input features: {exc}")

        try:
            prediction = model.predict(scaled_features)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model prediction failure: {exc}")

        label = "UNKNOWN"
        if isinstance(prediction, np.ndarray) and prediction.size > 0:
            pred_value = prediction[0]
            label = str(pred_value)
        else:
            label = str(prediction)

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(scaled_features)[0]
                confidence = float(np.max(proba))
            except Exception:
                confidence = 0.0

    # Map label to human readable name
    response_label = LABEL_MAP.get(str(label), str(label).upper())
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # Save to database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, prediction, confidence, features) VALUES (?, ?, ?, ?)",
            (timestamp, response_label, confidence, str(request.features))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")

    return PredictResponse(
        prediction=response_label,
        confidence=round(confidence, 4),
        timestamp=timestamp,
    )

@app.get("/simulate_traffic")
def simulate_traffic():
    normal_bounds = {
        "min": 0.0,
        "max": 1.0,
    }
    extreme_bounds = {
        "min": -10.0,
        "max": 10.0,
    }

    if np.random.rand() < 0.15:
        sample = np.random.uniform(extreme_bounds["min"], extreme_bounds["max"], 25)
    else:
        sample = np.random.uniform(normal_bounds["min"], normal_bounds["max"], 25)

    return {"features": sample.tolist(), "timestamp": datetime.utcnow().isoformat() + "Z"}

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "scaler_loaded": scaler is not None}
