# backend/src/training/train_delivery_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score # NEW: Regression metrics
import joblib
from pathlib import Path
from xgboost import XGBRegressor # NEW: Using the XGBoost Regressor

# Haversine distance function (calculates distance between lat/lng pairs)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def train_delivery_model():
    """
    Trains a regression model to predict the delivery duration of an order.
    """
    print("--- Starting Delivery Time Prediction Model Training ---")

    # --- 1. Define Paths ---
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODELS_PATH = PROJECT_ROOT / "models"
    
    # --- 2. Load and Prepare Data ---
    print("Step 1: Loading all necessary datasets...")
    try:
        orders = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_orders_cleaned_dataset.parquet')
        order_items = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_items_cleaned_dataset.parquet')
        products = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_products_cleaned_dataset.parquet')
        customers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_customers_cleaned_dataset.parquet')
        sellers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_sellers_cleaned_dataset.parquet')
        geolocation = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_geolocation_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure cleaned files are in '{PROCESSED_DATA_PATH}'.")
        return

    # Merge core datasets
    df = orders.merge(order_items, on='order_id').merge(products, on='product_id').merge(customers, on='customer_id').merge(sellers, on='seller_id')

    # --- 3. Feature Engineering ---
    print("Step 2: Engineering features for regression...")
    
    # NEW: Define the Target Variable for Regression
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['delivery_duration_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    # Geographic Distance Feature
    geo_avg = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    df = df.merge(geo_avg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}, inplace=True)
    df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
    df = df.merge(geo_avg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
    df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
    df['distance_km'] = haversine_distance(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])
    
    # --- 4. Final Data Preparation ---
    features = [
        'distance_km',
        'freight_value',
        'product_weight_g',
        'seller_state',
        'customer_state'
    ]
    target = 'delivery_duration_days'
    
    # Drop rows where the target or key features are missing
    final_df = df[features + [target]].dropna()
    # Ensure target is positive
    final_df = final_df[final_df[target] > 0]

    X = final_df[features]
    y = final_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 5. Create Preprocessing and Regression Pipeline ---
    print("Step 3: Building preprocessing and XGBoost Regressor pipeline...")
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # NEW: The pipeline uses XGBRegressor. No SMOTE or class weights needed for regression.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])

    # --- 6. Training ---
    print("Step 4: Training the regression model...")
    pipeline.fit(X_train, y_train)

    # --- 7. Evaluate and Save ---
    print("\nStep 5: Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Delivery Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"R-squared (R²): {r2:.2f}")
    print(f"\nInterpretation: On average, the model's prediction for delivery time is off by about {mae:.2f} days.")

    print("\nStep 6: Saving the delivery model pipeline...")
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PATH / 'delivery_model.joblib'
    joblib.dump(pipeline, model_path)
    print(f"✅ Delivery model saved successfully to {model_path}")
    
    print("\n--- Delivery Model Training Finished ---")

if __name__ == '__main__':
    train_delivery_model()