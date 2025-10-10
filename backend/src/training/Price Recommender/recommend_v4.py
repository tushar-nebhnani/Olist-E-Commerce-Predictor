import pandas as pd
import numpy as np
import joblib
import os
import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================================
# 0. CONFIGURATION AND FILE PATHS
# =========================================================================

# --- Directory and File Names ---
DATA_PATH = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\processed'
MODEL_DIR = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Price Recommender'
MODEL_NAME = 'xgboost_price_recommender_v4_log.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Input Parquet Files
PARQUET_FILES = {
    'orders': 'olist_orders_cleaned_dataset.parquet',
    'order_items': 'olist_order_items_cleaned_dataset.parquet',
    'reviews': 'olist_order_reviews_cleaned_dataset.parquet',
    'products': 'olist_products_cleaned_dataset.parquet',
    'category_translation': 'category_name_translation_cleaned_dataset.parquet',
}

# Best parameters found from the previous tuning step
BEST_XGBOOST_PARAMS = {
    'subsample': 0.6, 
    'n_estimators': 800, 
    'min_child_weight': 1, 
    'max_depth': 5, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.8
}

# =========================================================================
# 1. FEATURE ENGINEERING FUNCTION (V3 LOGIC)
# =========================================================================

def load_parquet_data(data_path, filename):
    """Helper to load parquet files."""
    full_path = os.path.join(data_path, filename)
    try:
        return pd.read_parquet(full_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {full_path}")
        return None

def engineer_master_dataset(data_path):
    """Loads and merges data, calculates high-impact features (Seller & Competition)."""
    
    print("--- Starting Feature Engineering (V4 Log Model) ---")
    
    df_map = {name: load_parquet_data(data_path, file) for name, file in PARQUET_FILES.items()}
    if any(df is None for df in df_map.values()):
        print("[FAILED] Cannot proceed due to missing input files.")
        return None

    # Base Merge: Orders, Items, Reviews (for delivered items only)
    df_orders = df_map['orders'][df_map['orders']['order_status'] == 'delivered'].copy()
    df_base = df_map['order_items'].merge(df_orders, on='order_id', how='left')
    
    df_reviews_agg = df_map['reviews'].groupby('order_id').first().reset_index()
    df_base = df_base.merge(df_reviews_agg[['order_id', 'review_score']], on='order_id', how='left')
    
    # Calculate New Features
    # 1. Seller Average Review Score (Seller Reputation)
    df_seller_review = df_base.groupby('seller_id')['review_score'].mean().reset_index().rename(columns={'review_score': 'seller_avg_review_score'})

    # 2. Product Competition Count
    df_competition = df_base.groupby('product_id')['seller_id'].nunique().reset_index().rename(columns={'seller_id': 'product_competition_count'})

    # Final Master Merge and Cleanup
    df_master = df_base.merge(df_seller_review, on='seller_id', how='left')
    df_master = df_master.merge(df_competition, on='product_id', how='left')
    df_master = df_master.merge(df_map['products'], on='product_id', how='left')
    df_master = df_master.merge(df_map['category_translation'][['product_category_name', 'product_category_name_english']], on='product_category_name', how='left')
    df_master['product_category_name_english'] = df_master['product_category_name_english'].fillna(df_master['product_category_name'])
    
    # 3. Price Per Volume (Quality proxy)
    df_master['product_volume_cm3'] = df_master['product_length_cm'] * df_master['product_height_cm'] * df_master['product_width_cm']
    epsilon = 1e-6 
    df_master['price_per_volume'] = df_master['price'] / (df_master['product_volume_cm3'] + epsilon)
    
    # Final cleanup and column selection
    FINAL_COLS = [
        'price', 'freight_value', 'review_score',
        'product_category_name_english', 'product_weight_g', 'product_length_cm', 
        'product_height_cm', 'product_width_cm', 'seller_avg_review_score', 
        'product_competition_count', 'price_per_volume'
    ]
    
    df_master = df_master[[col for col in FINAL_COLS if col in df_master.columns]].copy()
    
    # Drop NaNs on core features
    df_master.dropna(subset=['price', 'product_category_name_english', 'product_weight_g', 
                             'review_score', 'seller_avg_review_score'], inplace=True)
    
    # Remove zero/negative prices before log transform
    df_master = df_master[df_master['price'] > 0].copy()
    
    print(f"[SUCCESS] Final master data rows: {len(df_master):,}")
    return df_master

# =========================================================================
# 2. MAIN TRAINING AND INVERSE TRANSFORMATION
# =========================================================================

def train_final_log_model():
    df_master = engineer_master_dataset(DATA_PATH)
    if df_master is None:
        return

    # --- 2A. Log Transformation of Target ---
    TARGET = 'price'
    TARGET_LOG = 'price_log'
    df_master[TARGET_LOG] = np.log(df_master[TARGET])
    
    # Features (includes all engineered features)
    AVAILABLE_FEATURES = [col for col in df_master.columns if col not in [TARGET, TARGET_LOG, 'product_volume_cm3']]
    
    X = df_master[AVAILABLE_FEATURES]
    y_log = df_master[TARGET_LOG]
    X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    print("\n--- Starting XGBoost Training (V4 Log-Transformed) ---")

    # --- 2B. Preprocessing Setup ---
    numerical_features = [f for f in AVAILABLE_FEATURES if df_master[f].dtype in ['float64', 'int64']]
    categorical_features = ['product_category_name_english']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop' 
    )

    # --- 2C. XGBoost Pipeline with Best Parameters ---
    # XGBoost parameters are applied via **BEST_XGBOOST_PARAMS
    xgb_regressor = XGBRegressor(
        **BEST_XGBOOST_PARAMS,
        random_state=42, 
        n_jobs=-1, 
        eval_metric='rmse'
    )
    
    model_v4 = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb_regressor)
    ])

    model_v4.fit(X_train, y_log_train)
    
    # --- 2D. Evaluation (on Log and Inverse-Transformed Scale) ---
    y_log_pred = model_v4.predict(X_test)
    
    # Inverse transform predictions and actual values for R-squared and RMSE calculation
    y_test = np.exp(y_log_test)
    y_pred = np.exp(y_log_pred)
    
    # Calculate metrics on the original price scale
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Final XGBoost Model Performance (V4 Log) ---")
    print(f"Root Mean Squared Error (RMSE): R$ {rmse:,.2f} (on original scale)")
    print(f"R-squared (R²): {r2:.4f} (on original scale)")

    # --- 2E. Save the Final Model ---
    os.makedirs(MODEL_DIR, exist_ok=True) 
    joblib.dump(model_v4, MODEL_PATH)
    print(f"\n[SUCCESS] Final Model V4 saved to: '{MODEL_PATH}'")

    # --- 2F. Example Prediction Test ---
    loaded_model_v4 = joblib.load(MODEL_PATH)
    
    # Note: Example features must match the features used for training
    example_input = {
        'product_category_name_english': 'watches_gifts',
        'freight_value': 12.00,
        'product_weight_g': 350,
        'product_length_cm': 15,
        'product_height_cm': 5,
        'product_width_cm': 10,
        'review_score': 4.8,
        'seller_avg_review_score': 4.75, # Estimated seller reputation
        'product_competition_count': 5,  # Estimated competition level
        'price_per_volume': 0.8         # Estimated quality proxy
    }
    
    # Predict Log Price and then inverse transform
    y_log_pred_example = loaded_model_v4.predict(pd.DataFrame([example_input]))[0]
    recommended_price_final = np.exp(y_log_pred_example)

    print("\n--- Example Price Recommendation (Final V4 Log Model) ---")
    print(f"💰 Recommended Selling Price (V4): R$ {round(recommended_price_final, 2):,.2f}")

    return model_v4

# =========================================================================
# 3. API RECOMMENDATION FUNCTION
# =========================================================================

def recommend_price_api(input_data: dict, model_pipeline: Pipeline) -> float:
    """
    Generates final price recommendation using the log-transformed model. 
    This is the function your API endpoint will call.
    """
    new_data = pd.DataFrame([input_data])
    
    # 1. Predict the log-transformed price
    y_log_pred = model_pipeline.predict(new_data)[0]
    
    # 2. Inverse transform to get the dollar price
    predicted_price = np.exp(y_log_pred)
    
    return round(predicted_price, 2)

if __name__ == '__main__':
    train_final_log_model()