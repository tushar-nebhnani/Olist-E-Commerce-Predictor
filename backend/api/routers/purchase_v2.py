import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

# --- 1. Initialize and Load V2 Model ---
router = APIRouter()
logging.basicConfig(level=logging.INFO)

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = PROJECT_ROOT / "models" / "purchase_prediction" / "purchase_prediction_pipeline_v2.joblib"
    pipeline_v2 = joblib.load(MODEL_PATH)
    logging.info("✅ Purchase Prediction V2 model loaded successfully.")
except Exception as e:
    pipeline_v2 = None
    logging.warning(f"⚠️ Error loading Purchase Prediction V2 model: {e}")

# --- 2. Define the V2 Input Schema ---
# This includes all the advanced features the V2 model was trained on.
class PredictionFeaturesV2(BaseModel):
    # Original features
    price: float = 120.0
    freight_value: float = 20.0
    product_photos_qty: float = 2.0
    product_weight_g: float = 500.0
    product_volume_cm3: float = 1000.0
    distance_km: float = 500.0
    review_score: float = 4.0
    product_category_name_english: str = "health_beauty"
    customer_state: str = "SP"
    seller_state: str = "SP"
    
    # Advanced features
    customer_avg_review_score: float = 4.1
    customer_order_count: float = 1.0
    customer_total_spend: float = 150.0
    product_popularity: float = 50.0
    product_avg_review_score: float = 4.2
    price_vs_category_avg: float = -10.5
    category_popularity: float = 5000.0
    category_avg_review_score: float = 4.15

# --- 3. Create the V2 Prediction Endpoint ---
@router.post("/predict")
async def predict_purchase_v2(features: PredictionFeaturesV2):
    if pipeline_v2 is None:
        raise HTTPException(status_code=503, detail="Model v2 is not available or failed to load.")
    
    try:
        # Pydantic v2 uses model_dump(), v1 uses dict()
        try:
            input_data = features.model_dump()
        except AttributeError:
            input_data = features.dict()
            
        input_df = pd.DataFrame([input_data])
        
        prediction_proba = pipeline_v2.predict_proba(input_df)[:, 1][0]
        
        return {
            'model_version': 'v2 (Advanced)',
            'input_features': input_data,
            'purchase_probability': float(prediction_proba)
        }
    except Exception as e:
        logging.error(f"Prediction error with V2 model: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")
