# backend/api/routers/satisfaction_final.py

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

# --- 1. Initialize and Load the FINAL Model ---
router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "satisfaction_prediction" / "satisfaction_model_final.joblib"

pipeline_final = None
try:
    pipeline_final = joblib.load(MODEL_PATH)
    logging.info("✅ Optimised satisfaction model loaded successfully.")
except Exception as e:
    logging.info(f"⚠️ Error loading final model: {e}")

# --- 2. Define the FINAL Input Schema ---
# CHANGED: This now includes ALL features the final XGBoost model was trained on.
class PredictionFeaturesFinal(BaseModel):
    # Original V1 features
    price: float
    freight_value: float
    delivery_time_days: int
    estimated_vs_actual_delivery: int
    
    # Original V2 features
    payment_installments: int
    payment_value: float
    product_photos_qty: int
    product_weight_g: float
    product_category_name: str
    
    # Enhanced features for the final model
    seller_avg_review_score: float
    seller_order_count: int
    distance_km: float

# --- 3. Create the FINAL Prediction Endpoint ---
@router.post("/predict")
async def predict_satisfaction_final(features: PredictionFeaturesFinal):
    # CHANGED: Using the 'pipeline_final' variable
    if pipeline_final is None:
        raise HTTPException(status_code=503, detail="Final Model not loaded.")
    
    try:
        input_df = pd.DataFrame([features.dict()])
        
        # CHANGED: Using the 'pipeline_final' variable
        prediction = pipeline_final.predict(input_df)
        probability = pipeline_final.predict_proba(input_df)
        
        return {
            # CHANGED: Reporting the correct model version
            'model_version': 'final (XGBoost)',
            'is_satisfied_prediction': int(prediction[0]),
            'satisfaction_probability': float(probability[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")