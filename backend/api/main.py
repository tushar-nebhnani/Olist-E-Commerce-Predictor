# main.py - FINAL CORRECTED VERSION

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Clean up imports ---
# Import each router module directly
from .routers import satisfaction_final, satisfaction_v1, purchase_v1, purchase_v2, review_analysis_v1, product_recommendation_v1

# --- App Initialization ---
app = FastAPI(
    title="Olist Customer Satisfaction API",
    description="API for predicting customer satisfaction and product recommendations."
)

# --- 2. Add the Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """
    Loads all necessary ML models when the application starts.
    """
    print("--- Running application startup events ---")
    await product_recommendation_v1.load_recommendation_models(app)
    await review_analysis_v1.load_sentiment_model(app)
    print("--- Application startup events complete ---")

# --- Add Middleware ---
# Note: For production, you should restrict origins to your frontend's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Example: ["http://localhost:3000", "https://your-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Include All API Routers (with duplicate removed) ---
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