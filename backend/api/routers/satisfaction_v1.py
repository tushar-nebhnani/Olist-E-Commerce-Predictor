# backend/api/routers/satisfaction_v1.py

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path

# --- 1. Initialize and Load V1 Model ---
router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "satisfaction_model_v1.joblib"

pipeline_v1 = None
try:
    pipeline_v1 = joblib.load(MODEL_PATH)
    print("✅ Satisfaction V1 model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading V1 model: {e}")

# --- 2. Define the V1 Input Schema ---
class PredictionFeaturesV1(BaseModel):
    price: float = 129.90
    freight_value: float = 15.50
    delivery_time_days: int = 8
    estimated_vs_actual_delivery: int = 10

# --- 3. Create the V1 Prediction Endpoint ---
@router.post("/predict")
async def predict_satisfaction_v1(features: PredictionFeaturesV1):
    if pipeline_v1 is None:
        raise HTTPException(status_code=503, detail="V1 Model not loaded.")
    
    try:
        input_df = pd.DataFrame([features.dict()])
        prediction = pipeline_v1.predict(input_df)
        probability = pipeline_v1.predict_proba(input_df)
        
        return {
            'model_version': 'v1',
            'is_satisfied_prediction': int(prediction[0]),
            'satisfaction_probability': float(probability[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")