import pandas as pd
from pathlib import Path
import logging
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

# --- 1. CONFIGURATION AND SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define project paths
try:
    BASE_DIR = Path(__file__).resolve().parents[3]
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data/raw"
MODEL_OUTPUT_PATH = BASE_DIR / "models/Satisfaction Prediction"
MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
logging.info(f"Project base directory: {BASE_DIR}")
logging.info(f"Model artifacts will be saved to: {MODEL_OUTPUT_PATH}")

RANDOM_STATE = 42

# --- 2. DATA LOADING & FEATURE ENGINEERING ---

def load_and_engineer_features() -> pd.DataFrame:
    """Loads all necessary Olist datasets and engineers features for the V2 model."""
    logging.info("Loading all required datasets...")
    try:
        orders = pd.read_csv(DATA_DIR / 'olist_orders_dataset.csv')
        items = pd.read_csv(DATA_DIR / 'olist_order_items_dataset.csv')
        reviews = pd.read_csv(DATA_DIR / 'olist_order_reviews_dataset.csv')
        products = pd.read_csv(DATA_DIR / 'olist_products_dataset.csv')
        
        required_files = [orders, items, reviews, products]
        if any(df.empty for df in required_files):
            raise FileNotFoundError("One or more essential data files are empty or could not be read.")
            
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required data file was not found in '{DATA_DIR}'. Error: {e}")
        sys.exit(1)

    logging.info("Merging datasets...")
    # Merge datasets to create a comprehensive view
    df = orders.merge(items, on='order_id')
    df = df.merge(reviews, on='order_id')
    df = df.merge(products, on='product_id')

    logging.info("Engineering features...")
    
    # --- Convert date columns to datetime objects ---
    for col in ['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    # --- Create target variable: 'is_satisfied' ---
    df['is_satisfied'] = (df['review_score'] >= 4).astype(int)
    
    # --- Feature Engineering ---
    # Delivery time in days
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    # Difference between estimated and actual delivery
    df['estimated_vs_actual_delivery'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days

    # --- Select Features for V2 Model ---
    # These features match what the frontend sends
    feature_cols = [
        'price', 'freight_value', 'delivery_time_days', 'estimated_vs_actual_delivery',
        'payment_installments', 'product_photos_qty', 'product_weight_g'
    ]
    
    # Drop rows with missing values in key columns
    df.dropna(subset=feature_cols + ['is_satisfied'], inplace=True)

    logging.info(f"Feature engineering complete. Final dataframe shape: {df.shape}")
    
    # Return only the necessary columns for training
    return df[feature_cols + ['is_satisfied']]


# --- 3. MODEL TRAINING & SAVING ---

def train_and_save_satisfaction_model(df: pd.DataFrame):
    """Trains a LightGBM model and saves the entire preprocessing and model pipeline."""
    logging.info("Starting model training process...")

    X = df.drop('is_satisfied', axis=1)
    y = df['is_satisfied']

    # Identify numerical and categorical features (none in this V2 model, but good practice)
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Define the full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(random_state=RANDOM_STATE))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    
    logging.info("Training the LightGBM pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    # --- Evaluation ---
    logging.info("Evaluating model performance...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Satisfied', 'Satisfied'])
    
    logging.info(f"Model Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{report}")
    
    # --- Save the pipeline ---
    model_file = MODEL_OUTPUT_PATH / "satisfaction_model_v2.joblib"
    joblib.dump(model_pipeline, model_file)
    logging.info(f"Trained satisfaction pipeline saved to '{model_file}'")


# --- 4. MAIN ORCHESTRATOR ---
def main():
    """Main function to run the entire training pipeline."""
    logging.info("--- Starting V2 Satisfaction Prediction Training Pipeline ---")
    data = load_and_engineer_features()
    train_and_save_satisfaction_model(data)
    logging.info("--- V2 Satisfaction Prediction Training Pipeline Finished Successfully ---")

if __name__ == '__main__':
    main()
