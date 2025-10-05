# backend/src/training/train_delivery_model_v2.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path
from xgboost import XGBRegressor
import optuna  # NEW: Importing Optuna for advanced tuning

# Haversine distance function (no change)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def train_delivery_model_v2():
    """
    Trains an improved (V2) regression model using advanced features
    and powerful hyperparameter tuning with Optuna.
    """
    print("--- Starting Improved V2 Delivery Model Training ---")

    # --- 1. Load Data ---
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODELS_PATH = PROJECT_ROOT / "models"
    
    print("Step 1: Loading all necessary datasets...")
    try:
        orders = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_orders_cleaned_dataset.parquet')
        order_items = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_items_cleaned_dataset.parquet')
        products = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_products_cleaned_dataset.parquet')
        customers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_customers_cleaned_dataset.parquet')
        sellers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_sellers_cleaned_dataset.parquet')
        geolocation = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_geolocation_cleaned_dataset.parquet')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df = orders.merge(order_items, on='order_id').merge(products, on='product_id').merge(customers, on='customer_id').merge(sellers, on='seller_id')

    # --- 2. Advanced Feature Engineering ---
    print("Step 2: Engineering advanced features...")
    
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_delivered_carrier_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['delivery_duration_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    
    df['seller_shipping_time_days'] = (df['order_delivered_carrier_date'] - df['order_purchase_timestamp']).dt.days
    seller_shipping_stats = df.groupby('seller_id')['seller_shipping_time_days'].agg(['mean', 'std']).reset_index()
    seller_shipping_stats.columns = ['seller_id', 'seller_avg_shipping_time', 'seller_std_shipping_time']
    df = df.merge(seller_shipping_stats, on='seller_id', how='left')
    
    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    
    # NEW: is_same_state feature
    df['is_same_state'] = (df['customer_state'] == df['seller_state']).astype(int)

    geo_avg = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    df = df.merge(geo_avg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}, inplace=True)
    df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
    df = df.merge(geo_avg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
    df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
    df['distance_km'] = haversine_distance(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])
    
    # --- 3. Final Data Preparation ---
    features = [
        'distance_km', 'freight_value', 'product_weight_g', 'product_volume_cm3',
        'seller_avg_shipping_time', 'seller_std_shipping_time', # NEW: Added seller_std
        'purchase_day_of_week', 'purchase_month',
        'is_same_state', # NEW: Added is_same_state
        'seller_state', 'customer_state'
    ]
    target = 'delivery_duration_days'
    
    final_df = df[features + [target]].dropna()
    final_df = final_df[final_df[target] > 0]
    X = final_df[features]
    y = final_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 4. CHANGED: Define the Optuna Objective Function (replaces GridSearchCV) ---
    print("Step 3: Defining Optuna objective function...")
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        model = XGBRegressor(random_state=42, objective='reg:squarederror', **params)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
        return score.mean()

    # --- 5. CHANGED: Run the Optuna Study ---
    print("Step 4: Running Optuna hyperparameter search...")
    study = optuna.create_study(direction='maximize') # Maximize because 'neg_mean_absolute_error' is negative
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\nBest trial found:")
    print(f"  Value (Negative MAE): {study.best_value:.4f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # --- 6. Train Final Model and Evaluate ---
    print("\nStep 5: Training the final V2 model with best parameters...")
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror', **study.best_params))
    ])
    final_pipeline.fit(X_train, y_train)

    print("\nStep 6: Evaluating the final V2 model...")
    y_pred = final_pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Final Improved Delivery Model (V2) Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"R-squared (R²): {r2:.2f}")

    print("\nStep 7: Saving the final V2 delivery model...")
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PATH / 'delivery_model_v2.joblib' # Overwrites the previous V2 model
    joblib.dump(final_pipeline, model_path)
    print(f"✅ Final V2 delivery model saved successfully to {model_path}")

if __name__ == '__main__':
    train_delivery_model_v2()