import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path

# --- 1. Initialize and Load V2 Model ---
router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "satisfaction_model_v2.joblib"

pipeline_v2 = None
try:
    pipeline_v2 = joblib.load(MODEL_PATH)
    print("✅ Satisfaction V2 model loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading V2 model: {e}")

# --- 2. Define the V2 Input Schema ---
# This must match ALL features the V2 model was trained on
class PredictionFeaturesV2(BaseModel):
    price: float
    freight_value: float
    delivery_time_days: int
    estimated_vs_actual_delivery: int
    payment_installments: int
    payment_value: float
    product_photos_qty: int
    product_weight_g: float
    product_category_name: str

# --- 3. Create the V2 Prediction Endpoint ---
@router.post("/predict")
async def predict_satisfaction_v2(features: PredictionFeaturesV2):
    if pipeline_v2 is None:
        raise HTTPException(status_code=503, detail="V2 Model not loaded.")
    
    try:
        input_df = pd.DataFrame([features.dict()])
        prediction = pipeline_v2.predict(input_df)
        probability = pipeline_v2.predict_proba(input_df)
        
        return {
            'model_version': 'v2',
            'is_satisfied_prediction': int(prediction[0]),
            'satisfaction_probability': float(probability[0][1])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")