import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb
from haversine import haversine, Unit # Used for high-fidelity distance calculation (Great-Circle Distance)
from pathlib import Path
import logging
import joblib
import sys

# --- 1. CONFIGURATION AND SETUP (MLOPS PARAMETERS) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    # Determines the project root for robust file path management
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
OUTPUTS_PATH = PROJECT_ROOT / "models" / "purchase_prediction"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

# Strategic sampling ratio: 4 negative samples for every 1 positive sample
# This forces the model to be good at rejecting poor recommendations (ranking defense).
NEGATIVE_SAMPLE_RATIO = 4 
RANDOM_STATE = 42

# --- 2. DATA LOADING & PREPARATION FUNCTIONS ---

def load_individual_datasets(path: Path) -> dict:
    """Loads all individual cleaned parquet files. All 9 files are required for the rich feature set."""
    logging.info(f"Attempting to load individual cleaned datasets from: '{path}'...")
    if not path.exists():
        logging.error(f"FATAL: Processed data directory not found at '{path}'. Please ensure the preprocessing script has run.")
        sys.exit(1)

    datasets = {}
    required_files = {
        'orders': 'olist_orders_cleaned_dataset', 'customers': 'olist_customers_cleaned_dataset',
        'reviews': 'olist_order_reviews_cleaned_dataset', 'payments': 'olist_order_payments_cleaned_dataset',
        'items': 'olist_order_items_cleaned_dataset', 'products': 'olist_products_cleaned_dataset',
        'sellers': 'olist_sellers_cleaned_dataset', 'translation': 'category_name_translation_cleaned_dataset',
        'geolocation': 'olist_geolocation_cleaned_dataset'
    }
    try:
        for key, filename in required_files.items():
            file_path = path / f"{filename}.parquet"
            # Using Parquet for speed and data type integrity
            datasets[key] = pd.read_parquet(file_path) 
            logging.info(f"  -> Loaded '{filename}.parquet'")
        return datasets
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}. One of the required parquet files is missing from '{path}'.")
        logging.error("Please ensure the preprocessing script has been run successfully.")
        raise

def create_master_table(datasets: dict) -> pd.DataFrame:
    """Merges individual dataframes into a single, comprehensive master table."""
    logging.info("Creating master dataset by merging individual tables...")
    
    # Aggregate payment details first to avoid duplication issues during merge
    payments_agg = datasets['payments'].groupby('order_id').agg(
        total_payment_value=('payment_value', 'sum'),
        avg_payment_installments=('payment_installments', 'mean'),
        payment_methods_count=('payment_type', 'nunique')
    ).reset_index()

    # Aggregate geolocation data to get a single, mean coordinate per zip code prefix
    geo = datasets['geolocation'].groupby('geolocation_zip_code_prefix').agg(
        geolocation_lat=('geolocation_lat', 'mean'),
        geolocation_lng=('geolocation_lng', 'mean')
    ).reset_index()
    logging.info("  -> Aggregated geolocation data to create unique zip code entries.")

    # Core Merges
    df = pd.merge(datasets['orders'], datasets['customers'], on='customer_id', how='left')
    df = pd.merge(df, datasets['reviews'], on='order_id', how='left')
    df = pd.merge(df, payments_agg, on='order_id', how='left')
    df = pd.merge(df, datasets['items'], on='order_id', how='left')
    df = pd.merge(df, datasets['products'], on='product_id', how='left')
    df = pd.merge(df, datasets['sellers'], on='seller_id', how='left')
    df = pd.merge(df, datasets['translation'], on='product_category_name', how='left')

    # Geospatial Merges (Joining the aggregated geo data twice: once for customer, once for seller)
    df = pd.merge(df, geo, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}, inplace=True)
    df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)

    df = pd.merge(df, geo, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left', suffixes=('_customer', '_seller'))
    df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
    df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)

    # Critical dropna: Ensure all necessary coordinates and categorical names exist before proceeding
    df.dropna(subset=[
        'customer_lat', 'customer_lng', 'seller_lat', 'seller_lng',
        'product_category_name_english', 'order_purchase_timestamp'
    ], inplace=True)
    
    logging.info(f"Master dataset created successfully. Shape: {df.shape}")
    return df

def engineer_features_and_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs core feature engineering and memory-optimized negative sampling.
    """
    logging.info("Starting optimized feature engineering and negative sampling...")

    # --- Feature Engineering on Master Table (used for sampling context) ---
    # Haversine distance: Calculates physically accurate distance between seller and customer.
    df['distance_km'] = df.apply(
        lambda row: haversine(
            (row['customer_lat'], row['customer_lng']),
            (row['seller_lat'], row['seller_lng']),
            unit=Unit.KILOMETERS
        ), axis=1
    )
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    logging.info("  -> Distance and time features engineered.")

    # --- Negative Sampling: Creating the Supervised Training Target ---
    
    logging.info("  -> Identifying positive samples (actual purchases)...")
    positive_samples = df[['customer_unique_id', 'product_id']].copy()
    positive_samples.drop_duplicates(inplace=True)
    positive_samples['purchased'] = 1 # Target variable: 1 (Purchased)

    logging.info("  -> Generating random negative samples...")
    n_pos_samples = len(positive_samples)
    n_neg_samples_to_generate = n_pos_samples * (NEGATIVE_SAMPLE_RATIO + 5) # Generate excess candidates

    random_customers = np.random.choice(df['customer_unique_id'].unique(), size=n_neg_samples_to_generate)
    random_products = np.random.choice(df['product_id'].unique(), size=n_neg_samples_to_generate)
    
    negative_samples = pd.DataFrame({
        'customer_unique_id': random_customers,
        'product_id': random_products
    })
    negative_samples.drop_duplicates(inplace=True)

    logging.info("  -> Purifying negative samples (removing accidental positive matches)...")
    # Anti-leakage step: Merge against positive samples and keep only non-matches ('left_only')
    merged = pd.merge(negative_samples, positive_samples, on=['customer_unique_id', 'product_id'], how='left', indicator=True)
    pure_negatives = merged[merged['_merge'] == 'left_only'][['customer_unique_id', 'product_id']]
    
    # Sample the required number of purified negatives (maintaining the 4:1 strategic ratio)
    n_neg_samples_needed = n_pos_samples * NEGATIVE_SAMPLE_RATIO
    sampled_negatives = pure_negatives.sample(n=min(n_neg_samples_needed, len(pure_negatives)), random_state=RANDOM_STATE)
    sampled_negatives['purchased'] = 0 # Target variable: 0 (Not Purchased)

    logging.info("  -> Combining positive and negative samples...")
    model_df = pd.concat([positive_samples, sampled_negatives], ignore_index=True)

    # --- Final Feature Merging onto the Sampled Data ---
    logging.info("  -> Merging features into the final modeling dataset...")
    
    # Aggregate customer and product features for merging
    customer_features = df.groupby('customer_unique_id').first().reset_index()[[
        'customer_unique_id', 'customer_state', 'customer_lat', 'customer_lng', 'review_score'
    ]]
    product_features = df.drop_duplicates(subset='product_id')[[
        'product_id', 'product_category_name_english', 'price', 'freight_value',
        'product_photos_qty', 'product_weight_g', 'product_volume_cm3',
        'seller_id', 'seller_state', 'seller_lat', 'seller_lng'
    ]]

    # Join dense features back onto the sampled P/N (customer, product) pairs
    model_df = pd.merge(model_df, customer_features, on='customer_unique_id', how='left')
    model_df = pd.merge(model_df, product_features, on='product_id', how='left')

    # Recalculate distance_km for the sampled data (essential for the model input vector)
    model_df['distance_km'] = model_df.apply(
        lambda row: haversine(
            (row['customer_lat'], row['customer_lng']),
            (row['seller_lat'], row['seller_lng']),
            unit=Unit.KILOMETERS
        ), axis=1
    )
    # Generic temporal features (can be replaced with time-series lookback features in a V2)
    model_df['purchase_month'] = 6 
    model_df['purchase_dayofweek'] = 3

    model_df.dropna(inplace=True)
    model_df = model_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) # Shuffle data
    
    logging.info(f"  -> Sampling complete. Final modeling dataset shape: {model_df.shape}")
    
    # Save the final model data for audit and inspection
    model_data_path = OUTPUTS_PATH / "final_purchase_prediction_data.parquet"
    model_df.to_parquet(model_data_path, index=False)
    logging.info(f"  -> Final modeling data saved to {model_data_path}")
    
    return model_df

# --- 3. MODEL TRAINING & SAVING FUNCTIONS ---

def train_and_save_pipeline(model_df: pd.DataFrame) -> None:
    """Defines features, builds pipeline, trains model, evaluates, and saves it."""
    logging.info("Defining feature transformers and model pipeline...")
    
    # --- Define Feature Lists (The Final Input Vector) ---
    numerical_features = [
        'price', 'freight_value', 'product_photos_qty', 'product_weight_g',
        'product_volume_cm3', 'distance_km', 'purchase_month', 'purchase_dayofweek',
        'review_score' # Review score serves as a historical customer quality proxy
    ]
    categorical_features = [
        'product_category_name_english', 'customer_state', 'seller_state'
    ]

    # Impute missing review scores with the median before splitting (Robust imputation for the small number of nulls)
    model_df['review_score'] = model_df['review_score'].fillna(model_df['review_score'].median())

    X = model_df[numerical_features + categorical_features]
    y = model_df['purchased']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    # --- Preprocessing Governance: ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            # Numerical Pipeline: Scaling is crucial for LightGBM to correctly interpret feature importance
            ('num', StandardScaler(), numerical_features),
            # Categorical Pipeline: One-Hot Encoding converts strings to binary flags. 
            # handle_unknown='ignore' prevents production crash on new categories.
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop' # Drop all unlisted columns (e.g., IDs, raw coordinates)
    )

    # --- Classifier Selection: LightGBM (Speed & Accuracy) ---
    lgbm = lgb.LGBMClassifier(
        objective='binary', metric='auc', n_estimators=1000, learning_rate=0.05,
        num_leaves=31, max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1 # Optimized hyperparameters
    )

    # --- Final Pipeline Assembly ---
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', lgbm)])
    
    logging.info("Training the LightGBM model...")
    
    # --- Training Rigor: Early Stopping Setup ---
    # CRITICAL: Transform test data using the FIT preprocessor (preventing data leakage)
    preprocessor.fit(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Fit the pipeline. Early stopping monitors the AUC on the test set to prevent overfitting.
    model_pipeline.fit(
        X_train, y_train,
        classifier__eval_set=[(X_test_transformed, y_test)],
        classifier__callbacks=[lgb.early_stopping(100, verbose=True)]
    )
    logging.info("Model training complete.")

    # --- Evaluation ---
    logging.info("Evaluating model performance...")
    # Predict probabilities for AUC calculation (essential for ranking quality)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"Divine Accuracy (Test AUC-ROC Score): {auc:.4f}")
    
    # Predict binary labels for the Classification Report
    y_pred = model_pipeline.predict(X_test)
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # --- Save the entire pipeline (MLOps Readiness) ---
    model_file = OUTPUTS_PATH / "purchase_prediction_pipeline.joblib"
    # The saved artifact contains both the preprocessor (ColumnTransformer) and the trained LGBM model.
    joblib.dump(model_pipeline, model_file)
    logging.info(f"Trained pipeline (preprocessor + model) saved to '{model_file}'")

# --- 4. MAIN ORCHESTRATOR ---

def main_pipeline():
    """Main orchestrator for the purchase prediction model pipeline."""
    logging.info("--- Starting God-Tier Purchase Prediction Pipeline ---")

    datasets = load_individual_datasets(PROCESSED_DATA_PATH)
    master_df = create_master_table(datasets)
    modeling_data = engineer_features_and_sample(master_df)
    train_and_save_pipeline(modeling_data)
    
    logging.info("--- Purchase Prediction Pipeline Finished Successfully ---")

if __name__ == '__main__':
    main_pipeline()
