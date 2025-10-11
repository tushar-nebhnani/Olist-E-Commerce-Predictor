# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# --- 1. Standardize imports and configure logging ---
# This helps ensure modules are found correctly and provides startup feedback.
from backend.api.routers import (
    satisfaction_final, 
    satisfaction_v1, 
    purchase_v1, 
    purchase_v2, 
    review_analysis_v1, 
    product_recommendation_v1, 
    review_analysis_v2
)

logging.basicConfig(level=logging.INFO)


# --- App Initialization ---
app = FastAPI(
    title="Olist Customer Satisfaction API",
    description="API for predicting customer satisfaction and product recommendations."
)

# --- 2. Add a more robust Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """
    Loads all necessary ML models when the application starts and verifies them.
    """
    logging.info("--- 🚀 Running application startup events ---")
    
    # Load all models
    await product_recommendation_v1.load_recommendation_models(app)
    await review_analysis_v1.load_sentiment_model(app)
    
    logging.info("--- ✅ Application startup events complete ---")

    # FIX: Add a verification step to confirm the model is in memory.
    # This directly addresses the "AttributeError: 'State' object has no attribute 'sentiment_model'"
    if hasattr(app.state, 'sentiment_model') and hasattr(app.state, 'sentiment_vectorizer'):
        logging.info("👍 Sentiment model and vectorizer were loaded into app.state successfully.")
    else:
        logging.error("🔥 CRITICAL ERROR: Sentiment model or vectorizer FAILED to load into app.state. Check the file paths in the loading functions.")


# --- Add Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Include All API Routers ---
app.include_router(
    satisfaction_v1.router,
    prefix="/satisfaction/v1",
    tags=["Satisfaction Prediction V1 (Baseline)"]
)
app.include_router(
    satisfaction_final.router,
    prefix="/satisfaction/final",
    tags=["Satisfaction Prediction FINAL (XGBoost)"]
)
app.include_router(
    review_analysis_v1.router,
    prefix="/review/v1",
    tags=["Review Analysis"]
)
app.include_router(
    review_analysis_v2.router, 
    prefix="/api/v2", 
    tags=["Review Analysis V2"]
)
app.include_router(
    purchase_v1.router,
    prefix="/purchase/v1",
    tags=["Purchase Prediction V1 (Baseline)"]
)
app.include_router(
    purchase_v2.router,
    prefix="/purchase/v2",
    tags=["Purchase Prediction V2 (Advanced)"]
)
app.include_router(
    product_recommendation_v1.router,
    prefix="/recommendation",
    tags=["Product Recommendation"]
)

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {
        "status": "API is running.",
        "docs_url": "/docs"
    }