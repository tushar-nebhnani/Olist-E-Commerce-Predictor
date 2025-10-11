# main.py - CORRECTED VERSION

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import all your routers
from .routers import satisfaction_final, satisfaction_v1, purchase_v1, purchase_v2

from .routers.review_analysis_v1 import (
    load_sentiment_model,
    router as sentiment_router
)

from .routers.product_recommendation_v1 import (
    load_recommendation_models,
    router as recommendation_router
)

# --- App Initialization ---
app = FastAPI(
    title="Olist Customer Satisfaction API",
    description="API for predicting customer satisfaction and product recommendations."
)

# --- 2. ADD THE STARTUP EVENT HANDLER ---
# This decorator tells FastAPI to run this function once, right after it starts up.
@app.on_event("startup")
async def startup_event():
    """
    Loads all necessary ML models when the application starts.
    """
    print("--- Running application startup events ---")
    # This line is the critical fix. It calls the function to load your models.
    await load_recommendation_models(app)
    await load_sentiment_model(app) 

    print("--- Application startup events complete ---")


# --- Add Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include All API Routers ---
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
    purchase_v1.router,
    prefix="/purchase/v1",
    tags=["Purchase Prediction V1 (Baseline)"]
)
app.include_router(
    purchase_v2.router,
    prefix="/purchase/v2",
    tags=["Purchase Prediction V2 (Advanced)"]
)
# --- 3. INCLUDE THE RECOMMENDATION ROUTER ---
app.include_router(
    recommendation_router,
    prefix="/recommendation",
    tags=["Product Recommendation"]
)

app.include_router(
    sentiment_router,
    prefix="/sentiment",
    tags=["Review Analysis"]
)

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {
        "status": "API is running.",
        "docs_url": "/docs"
    }