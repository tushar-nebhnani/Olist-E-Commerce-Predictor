import pandas as pd
import re
from pathlib import Path
import logging
import joblib
import sys
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --- 1. CONFIGURATION AND SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # Dynamic path resolution for model outputs
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # Fallback for interactive environments
    PROJECT_ROOT = Path.cwd()

logging.info(f"Project root automatically set to: {PROJECT_ROOT}")

# Using a specific, hardcoded path for the model output as requested.
# The 'r' prefix creates a raw string to handle backslashes correctly on Windows.
MODEL_OUTPUT_PATH = Path(r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Review Analysis")
MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logging.info(f"Model artifacts will be saved to: {MODEL_OUTPUT_PATH}")

RANDOM_STATE = 42

# --- 2. DATA LOADING & PREPARATION FUNCTIONS ---

def download_nltk_resources():
    """Checks for and downloads all necessary NLTK resources for the script."""
    logging.info("Checking for required NLTK resources...")
    resources = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
    }
    for resource_id, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
            logging.info(f" -> NLTK resource '{resource_id}' already downloaded.")
        except LookupError:
            logging.info(f" -> Downloading NLTK resource '{resource_id}'...")
            nltk.download(resource_id, quiet=True)

def load_and_prepare_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the raw review data, creates binary sentiment labels,
    and drops irrelevant entries.
    """
    logging.info(f"Loading and preparing data from '{filepath}'...")
    if not filepath.exists():
        logging.error(f"FATAL: Data file not found at '{filepath}'.")
        sys.exit(1)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"Failed to load data file: {e}")
        raise

    df.dropna(subset=['review_comment_message', 'review_score'], inplace=True)

    def to_sentiment(rating):
        if rating <= 2: return 0  # Negative
        if rating >= 4: return 1  # Positive
        return None  # Neutral

    df['sentiment'] = df['review_score'].apply(to_sentiment)
    df.dropna(subset=['sentiment'], inplace=True)
    df['sentiment'] = df['sentiment'].astype(int)
    
    logging.info(f"Data prepared. Shape: {df.shape}. Sentiment distribution:\n{df['sentiment'].value_counts(normalize=True)}")
    return df[['review_comment_message', 'sentiment']]

def preprocess_text(text: str) -> str:
    """Cleans and preprocesses a single text string for sentiment analysis."""
    # Lazy initialization of stopwords to avoid loading them for every call
    if not hasattr(preprocess_text, "stop_words"):
        stop_words_pt = set(stopwords.words('portuguese'))
        negation_words = {'não', 'nem', 'nunca'}
        preprocess_text.stop_words = stop_words_pt - negation_words
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I | re.A)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in preprocess_text.stop_words]
    
    return " ".join(filtered_tokens)

# --- 3. MODEL TRAINING & SAVING FUNCTIONS ---

def train_and_save_pipeline(df: pd.DataFrame) -> None:
    """
    Defines features, builds a pipeline with a vectorizer and a classifier,
    trains the model, evaluates it, and saves the final pipeline object.
    """
    logging.info("Defining feature transformers and model pipeline...")

    X = df['cleaned_message']
    y = df['sentiment']

    # Split data, stratifying by y to maintain sentiment ratio in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    logging.info(f"Data split. Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    # Create a full pipeline that includes vectorization and classification
    model_pipeline = Pipeline(steps=[
        ('vectorizer', TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 2) # Consider both single words and word pairs
        )),
        ('classifier', LogisticRegression(
            max_iter=1000, 
            random_state=RANDOM_STATE,
            solver='liblinear' # Good for smaller datasets
        ))
    ])
    
    logging.info("Training the sentiment analysis pipeline...")
    model_pipeline.fit(X_train, y_train)
    logging.info("Model training complete.")

    # --- Evaluation ---
    logging.info("Evaluating model performance on the test set...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    
    logging.info(f"Model Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{report}")

    # --- Save the entire pipeline ---
    model_file = MODEL_OUTPUT_PATH / "sentiment_analysis_pipeline.joblib"
    joblib.dump(model_pipeline, model_file)
    logging.info(f"Trained pipeline (vectorizer + model) saved to '{model_file}'")

# --- 4. MAIN ORCHESTRATOR ---

def main_pipeline():
    """Main orchestrator for the sentiment analysis model training pipeline."""
    logging.info("--- Starting Sentiment Analysis Training Pipeline ---")
    
    download_nltk_resources()
    
    # --- FIX: Using a specific, hardcoded path for the dataset ---
    # The 'r' prefix creates a raw string to handle backslashes correctly on Windows.
    raw_data_file = Path(r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\raw\olist_order_reviews_dataset.csv")
    
    data = load_and_prepare_data(raw_data_file)
    
    logging.info("Applying text preprocessing to all review messages...")
    data['cleaned_message'] = data['review_comment_message'].apply(preprocess_text)
    
    train_and_save_pipeline(data)
    
    logging.info("--- Sentiment Analysis Pipeline Finished Successfully ---")

if __name__ == '__main__':
    main_pipeline()

