# recommendation_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import logging
from pathlib import Path

# --- 1. Model Setup ---
router = APIRouter(prefix="/recommend", tags=["Recommendations"])
logging.basicConfig(level=logging.INFO)

# Define the expected structure of a single recommendation from the model output
class RecommendedProduct(BaseModel):
    product_id: str
    product_category_name: str
    price: float

# Define the expected response structure
class RecommendationResponse(BaseModel):
    customer_id: str
    recommended_products: list[RecommendedProduct]

# --- Model Loading (Placeholder Logic) ---
# NOTE: Replace this path logic with your actual model file location
try:
    # Assuming the router is run from the 'backend' root or similar path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODEL_PATH = PROJECT_ROOT / "models" / "recommendation" / "your_rec_model.joblib"
    
    # Placeholder: Assuming a large recommendation matrix or model
    # rec_model = joblib.load(MODEL_PATH) 
    rec_model = "Mock_Recommendation_Model" 
    logging.info("✅ Recommendation model loaded successfully (Mocked).")
except Exception as e:
    rec_model = None
    logging.warning(f"⚠️ Error loading Recommendation model: {e}")
    
# --- Mock Data Source (Replace with your database/data access) ---
MOCK_CUSTOMER_DATA = {
    "abc1234567890": [
        {"product_id": "prod_1", "product_category_name": "health_beauty", "price": 125.50},
        {"product_id": "prod_2", "product_category_name": "computers_accessories", "price": 599.99},
        {"product_id": "prod_3", "product_category_name": "watches_gifts", "price": 79.00},
    ],
    "def9876543210": [
        {"product_id": "prod_4", "product_category_name": "housewares", "price": 45.90},
    ]
}

# --- 2. Prediction Endpoint ---
@router.get("/products/{customer_id}", response_model=RecommendationResponse)
async def get_product_recommendations(customer_id: str):
    if rec_model is None:
        raise HTTPException(status_code=503, detail="Recommendation service is currently unavailable.")
    
    # --- Actual Recommendation Logic Placeholder ---
    
    # 1. Look up recommendations for the customer
    recommendations = MOCK_CUSTOMER_DATA.get(customer_id.lower())

    if not recommendations:
        # If customer ID is known but has no recommendations, return 200 with empty list
        if customer_id.lower() in MOCK_CUSTOMER_DATA:
             return RecommendationResponse(customer_id=customer_id, recommended_products=[])
             
        # If customer ID is completely unknown, raise 404
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found in the recommendation system.")
    
    # 2. Return the results
    return RecommendationResponse(
        customer_id=customer_id,
        recommended_products=recommendations
    )