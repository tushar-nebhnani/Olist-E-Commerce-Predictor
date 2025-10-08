from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import satisfaction_final, satisfaction_v1, purchase_v1, purchase_v2

# --- App Initialization ---
app = FastAPI(
    title="Olist Customer Satisfaction API",
    description="API for predicting customer satisfaction. Includes a baseline V1 and an improved V2 model."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include API Routers ---
# Each router handles a specific version or model type.
# This makes the API clean and easy to scale.
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

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {
        "status": "API is running.",
        "docs_url": "/docs"
    }