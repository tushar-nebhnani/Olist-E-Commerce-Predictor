import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

# --- 1. Initialize and Load V1 Model ---
router = APIRouter()
logging.basicConfig(level=logging.INFO)

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = PROJECT_ROOT / "models" / "purchase_prediction" / "purchase_prediction_pipeline.joblib"
    pipeline_v1 = joblib.load(MODEL_PATH)
    logging.info("✅ Purchase Prediction V1 model loaded successfully.")
except Exception as e:
    pipeline_v1 = None
    logging.warning(f"⚠️ Error loading Purchase Prediction V1 model: {e}")

# --- 2. Define the V1 Input Schema ---
# These are the features the baseline model was trained on.
class PredictionFeaturesV1(BaseModel):
    price: float = 120.0
    freight_value: float = 20.0
    product_photos_qty: float = 2.0
    product_weight_g: float = 500.0
    product_volume_cm3: float = 1000.0
    distance_km: float = 500.0
    purchase_month: int = 6
    purchase_dayofweek: int = 3
    product_category_name_english: str = "health_beauty"
    customer_state: str = "SP"
    seller_state: str = "SP"
    review_score: float = 4.0

# --- 3. Create the V1 Prediction Endpoint ---
@router.post("/predict")
async def predict_purchase_v1(features: PredictionFeaturesV1):
    if pipeline_v1 is None:
        raise HTTPException(status_code=503, detail="Model v1 is not available or failed to load.")
    
    try:
        input_df = pd.DataFrame([features.dict()])
        
        # Predict the probability of the positive class (1 = purchased)
        prediction_proba = pipeline_v1.predict_proba(input_df)[:, 1][0]
        
        return {
            'model_version': 'v1 (Baseline)',
            'input_features': features.dict(),
            'purchase_probability': float(prediction_proba)
        }
    except Exception as e:
        logging.error(f"Prediction error with V1 model: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")
