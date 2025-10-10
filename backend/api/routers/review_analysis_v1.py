# File: backend/api/routers/review_analysis_v1.py

import os
import joblib
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi import APIRouter, Request, HTTPException, FastAPI
from pydantic import BaseModel

# --- 1. SETUP AND CONFIGURATION ---
router = APIRouter()

# Pydantic model for input validation
class ReviewInput(BaseModel):
    review_text: str

# --- 2. PREPROCESSING LOGIC (Copied from your training script) ---
# This ensures that new data is processed exactly like the training data.

def download_nltk_resources():
    """Checks for and downloads all necessary NLTK resources."""
    resources = {'stopwords': 'corpora/stopwords', 'punkt': 'tokenizers/punkt'}
    for resource_id, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            logging.info(f"NLTK resource '{resource_id}' found.")
        except LookupError:
            logging.info(f"Downloading NLTK resource '{resource_id}'...")
            nltk.download(resource_id, quiet=True)

def preprocess_text(text: str) -> str:
    """Cleans and preprocesses a single text string for sentiment analysis."""
    if not hasattr(preprocess_text, "stop_words"):
        stop_words_pt = set(stopwords.words('portuguese'))
        negation_words = {'não', 'nem', 'nunca'}
        preprocess_text.stop_words = stop_words_pt - negation_words
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I | re.A)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in preprocess_text.stop_words]
    
    return " ".join(filtered_tokens)

# --- 3. MODEL LOADING ON STARTUP ---

async def load_sentiment_model(app: FastAPI):
    """
    Loads the sentiment analysis pipeline and NLTK resources once on startup
    and attaches them to the application state.
    """
    logging.info("--- Loading Sentiment Analysis Model ---")
    try:
        # Define path to the saved model artifact
        ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
        API_DIR = os.path.dirname(ROUTER_DIR)
        BACKEND_DIR = os.path.dirname(API_DIR)
        MODEL_PATH = os.path.join(BACKEND_DIR, "models", "Review Analysis", "sentiment_analysis_pipeline.joblib")
        
        logging.info(f"Model path: {MODEL_PATH}")

        # Ensure NLTK resources are available
        download_nltk_resources()

        # Load the trained pipeline
        app.state.sentiment_pipeline = joblib.load(MODEL_PATH)
        logging.info("✅ Sentiment analysis pipeline loaded successfully!")
        app.state.sentiment_model_ready = True
    except Exception as e:
        logging.error(f"❌ Failed to load sentiment model: {e}", exc_info=True)
        app.state.sentiment_model_ready = False

def check_model_loaded(request: Request):
    """Helper function to protect endpoint if model loading failed."""
    if not getattr(request.app.state, 'sentiment_model_ready', False):
        raise HTTPException(status_code=503, detail="Service unavailable: Sentiment model not loaded.")

# --- 4. API ENDPOINT FOR PREDICTION ---

@router.post("/predict", tags=["Review Analysis"])
async def predict_sentiment(review: ReviewInput, request: Request):
    """
    Predicts the sentiment of a given review text.
    - Input: {"review_text": "your review here"}
    - Output: {"sentiment": "Positive/Negative", "sentiment_score": 0.95}
    """
    check_model_loaded(request)
    
    try:
        pipeline = request.app.state.sentiment_pipeline
        original_text = review.review_text
        
        # 1. Preprocess the input text
        cleaned_text = preprocess_text(original_text)
        
        # 2. Predict the sentiment (model expects a list of texts)
        prediction = pipeline.predict([cleaned_text])
        probabilities = pipeline.predict_proba([cleaned_text])
        
        sentiment_code = int(prediction[0])
        sentiment_score = float(probabilities[0][sentiment_code])
        sentiment_label = "Positive" if sentiment_code == 1 else "Negative"

        return {
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "sentiment": sentiment_label,
            "sentiment_score": round(sentiment_score, 4),
            "prediction": sentiment_code
        }
    except Exception as e:
        logging.error(f"Error during sentiment prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during prediction.")