from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging

# Local application imports
from .review_analysis_v1 import preprocess_text # Re-use the same preprocessing logic

# --- Router Configuration ---
router = APIRouter()

# --- Pydantic Models for V2 ---
class ReviewRequestV2(BaseModel):
    """Request model for V2, allowing optional return of cleaned text."""
    text: str
    include_cleaned_text: bool = False

class SentimentResponseV2(BaseModel):
    """Response model for V2, with optional cleaned text field."""
    text: str
    sentiment: str
    probability: float
    cleaned_text: str | None = None

# --- API Endpoint for V2 ---
@router.post("/sentiment", response_model=SentimentResponseV2)
async def predict_sentiment_v2(payload: ReviewRequestV2, request: Request):
    """
    Analyzes the sentiment of a given review text (V2).

    This version introduces an optional feature to return the preprocessed text.

    - **text**: The customer review you want to analyze.
    - **include_cleaned_text**: Set to `true` to include the cleaned text in the response.
    """
    # Access the globally loaded model and vectorizer from the app's state
    model = request.app.state.sentiment_model
    vectorizer = request.app.state.sentiment_vectorizer

    if not model or not vectorizer:
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Sentiment analysis model is not available. Please check server logs."
        )

    try:
        # 1. Preprocess the input text using the shared function
        cleaned_text = preprocess_text(payload.text)

        # 2. Vectorize the cleaned text
        vectorized_text = vectorizer.transform([cleaned_text])

        # 3. Predict sentiment and probability
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text).max()
        sentiment = "Positive" if prediction == 1 else "Negative"

        # 4. Construct the response
        response_data = {
            "text": payload.text,
            "sentiment": sentiment,
            "probability": float(probability)
        }
        if payload.include_cleaned_text:
            response_data["cleaned_text"] = cleaned_text

        return SentimentResponseV2(**response_data)

    except Exception as e:
        logging.error(f"Error during V2 sentiment prediction: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
