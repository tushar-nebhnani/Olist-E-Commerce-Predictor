# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Import all your routers ---
from .routers import satisfaction_v1, satisfaction_v2

# --- App Initialization ---
app = FastAPI(
    title="Olist Customer Satisfaction API",
    description="API for predicting customer satisfaction. Includes a baseline V1 and an improved V2 model."
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"], 
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
    satisfaction_v2.router, 
    prefix="/satisfaction/v2", 
    tags=["Satisfaction Prediction V2 (Improved)"]
)

# --- Root Endpoint ---
@app.get("/")
def read_root():
    return {
        "status": "API is running.",
        "docs_url": "/docs"
    }