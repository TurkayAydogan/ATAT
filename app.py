import pandas as pd
import random
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist

# 1. Modelleri ve Test Verisi
app = FastAPI()

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

# Saldırı İsimleri Sözlüğü 
CLASS_NAMES = {
    0: "BENIGN (GÜVENLİ)", 1: "Bot", 2: "DDoS", 3: "DoS GoldenEye", 4: "DoS Hulk", 
    5: "DoS Slowhttptest", 6: "DoS slowloris", 7: "FTP-Patator", 
    8: "Heartbleed", 9: "Infiltration", 10: "PortScan", 
    11: "SSH-Patator", 12: "Web Attack - Brute Force", 
    13: "Web Attack - Sql Injection", 14: "Web Attack - XSS"
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
    
    return {
        "prediction": CLASS_NAMES.get(pred_index, "Bilinmeyen"),
        "confidence": real_confidence
    }