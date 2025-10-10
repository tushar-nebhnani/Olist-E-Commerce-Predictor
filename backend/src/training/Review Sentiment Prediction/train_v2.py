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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler

# --- 1. CONFIGURATION AND SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Using the specific, hardcoded path for the model output as requested.
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
    Vectorizes data, applies under-sampling, performs hyperparameter tuning,
    evaluates the best model, and saves the final vectorizer and classifier.
    """
    logging.info("Defining feature transformers and model pipeline...")

    df['cleaned_message'] = df['review_comment_message'].apply(preprocess_text)
    X = df['cleaned_message']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    logging.info(f"Data split. Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    logging.info("Fitting vectorizer on training data...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logging.info("Applying Random Under-Sampling to the training set...")
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_vec, y_train)
    logging.info(f"Training data resampled. New shape: {X_train_resampled.shape}")

    # --- HYPERPARAMETER TUNING ---
    logging.info("Starting hyperparameter tuning for LightGBM...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 40, 50],
    }
    lgbm = lgb.LGBMClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=lgbm, param_grid=param_grid, scoring='f1_weighted', cv=3, n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train_resampled, y_train_resampled)
    logging.info(f"Hyperparameter tuning complete. Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_

    # --- Evaluation ---
    logging.info("Evaluating best model performance on the original (unbalanced) test set...")
    y_pred = best_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    logging.info(f"Model Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{report}")

    # --- Save the final pipeline object ---
    pipeline_to_save = {
        'vectorizer': vectorizer,
        'model': best_model
    }
    # Using the new, more descriptive model name
    model_file = MODEL_OUTPUT_PATH / "advanced_sentiment_pipeline.joblib"
    joblib.dump(pipeline_to_save, model_file)
    logging.info(f"Trained pipeline (vectorizer + model) saved to '{model_file}'")

# --- 4. MAIN ORCHESTRATOR ---

def main_pipeline():
    """Main orchestrator for the sentiment analysis model training pipeline."""
    logging.info("--- Starting Advanced Sentiment Analysis Training Pipeline ---")
    download_nltk_resources()
    raw_data_file = Path(r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\raw\olist_order_reviews_dataset.csv")
    data = load_and_prepare_data(raw_data_file)
    train_and_save_pipeline(data)
    logging.info("--- Sentiment Analysis Pipeline Finished Successfully ---")

if __name__ == '__main__':
    main_pipeline()

