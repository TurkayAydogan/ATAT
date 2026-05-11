import pandas as pd
import random
import joblib
import numpy as np
import sqlite3
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist

# 1. Modelleri ve Test Verisi
app = FastAPI()

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
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

@app.on_event("startup")
def startup_event():
    init_db()

# Frontend ile bağlantı için CORS ayarı
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model ve Scaler dosyaları
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Gerçek test verisi
try:
    test_df = pd.read_csv('test_veri_akisi.csv')
    print("✅ Gerçek test verisi başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Hata: CSV yüklenemedi! {e}")
    test_df = None

# Label Mapping in English
CLASS_NAMES = {
    0: "BENIGN", 
    1: "Bot", 
    2: "DDoS", 
    3: "DoS GoldenEye", 
    4: "DoS Hulk", 
    5: "DoS Slowhttptest", 
    6: "DoS Slowloris", 
    7: "FTP Brute Force", 
    8: "Heartbleed", 
    9: "Infiltration", 
    10: "Port Scan", 
    11: "SSH Brute Force", 
    12: "Web Attack - Brute Force", 
    13: "Web Attack - SQL Injection", 
    14: "Web Attack - XSS"
}

class PredictRequest(BaseModel):
    features: conlist(float, min_length=25, max_length=25)

@app.get("/simulate_traffic")
async def simulate_traffic():
    """CSV içinden rastgele gerçek bir paket seçer."""
    if test_df is not None:
        random_index = random.randint(0, len(test_df) - 1)
        row = test_df.iloc[random_index]
        
        # 'GERCEK_ETIKET' hariç tüm özellikleri al
        features = row.drop('GERCEK_ETIKET').tolist()
        return {"features": features}
    return {"error": "CSV dosyası yüklenemedi"}

@app.post("/predict")
async def predict(request: PredictRequest):
    """Gelen paketi model ile analiz eder ve gerçek olasılık skorunu hesaplar."""
    features_array = np.array(request.features).reshape(1, -1)
    
    # Önce ölçekle 
    scaled_features = scaler.transform(features_array)
    
    # Tahmin edilen sınıfı bul 
    pred_index = int(model.predict(scaled_features)[0])
    
    # YAPAY ZEKANIN GERÇEK GÜVEN SKORUNU HESAPLA 
    # Modelin o sınıf için yüzde kaç ihtimal verdiğini çekiyoruz
    probabilities = model.predict_proba(scaled_features)[0]
    real_confidence = float(probabilities[pred_index])
    
    prediction = CLASS_NAMES.get(pred_index, "Unknown")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to Database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, prediction, confidence) VALUES (?, ?, ?)",
            (timestamp, prediction, real_confidence)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

    return {
        "prediction": prediction,
        "confidence": real_confidence
    }

@app.get("/history")
async def get_history():
    """Veritabanındaki son 50 kaydı getirir."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, prediction, confidence FROM logs ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "timestamp": row[0],
                "prediction": row[1],
                "confidence": row[2]
            })
        return history
    except Exception as e:
        return {"error": str(e)}
