from datetime import datetime
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, conlist

app = FastAPI(title="Real-Time SOC Dashboard API")

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
def load_artifacts():
    global model, scaler, class_labels

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{MODEL_PATH}': {exc}")

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load scaler '{SCALER_PATH}': {exc}")

    if hasattr(model, "classes_"):
        class_labels = [str(label) for label in model.classes_]
    elif hasattr(model, "get_booster"):
        class_labels = ["BENIGN", "ATTACK"]
    else:
        class_labels = ["BENIGN", "ATTACK"]


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler is not loaded")

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

    if label.upper() == "BENIGN":
        response_label = "BENIGN"
    else:
        response_label = str(label).upper()

    return PredictResponse(
        prediction=response_label,
        confidence=round(confidence, 4),
        timestamp=datetime.utcnow().isoformat() + "Z",
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
