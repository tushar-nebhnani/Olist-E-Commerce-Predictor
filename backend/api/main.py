from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Initialize the FastAPI app
app = FastAPI(title="Olist Customer Satisfaction API")

# Configure CORS to allow requests from your React frontend
origins = [
    "http://localhost:5173", # For Vite
    "http://localhost:3000", # For Create React App
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust if your frontend runs on a different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model pipeline
# The path is relative to where you run the 'uvicorn' command (the 'backend' directory)
MODEL_PATH = os.path.join("models", "satisfaction_model_v1.joblib")
try:
    pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    pipeline = None

# Define the data structure for incoming requests
class PredictionFeatures(BaseModel):
    price: float = 129.90
    freight_value: float = 15.50
    delivery_time_days: int = 8
    estimated_vs_actual_delivery: int = 10


# Create the /predict endpoint
@app.post("/predict_satisfaction")
async def predict_satisfaction(features: PredictionFeatures):
    if not pipeline:
        return {"error": "Model not loaded. Please train the model first."}

    try:
        # Convert Pydantic model to a DataFrame that the pipeline expects
        data_df = pd.DataFrame([features.dict()])

        # Make prediction and get probability
        prediction = pipeline.predict(data_df)
        probability = pipeline.predict_proba(data_df)

        # Prepare the response
        result = {
            'prediction_is_satisfied': int(prediction[0]),
            'probability_is_satisfied': float(probability[0][1])
        }
        return result

    except Exception as e:
        return {"error": str(e)}

# Root endpoint for a basic health check
@app.get("/")
def read_root():
    return {"status": "API is running."}