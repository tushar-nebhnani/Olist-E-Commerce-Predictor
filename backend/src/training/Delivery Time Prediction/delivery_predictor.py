import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth
    given their latitude and longitude.
    """
    R = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def run_delivery_prediction_pipeline(data_directory, model_save_path):
    """
    Loads data, engineers features, tunes a model, prints a full report,
    and saves the final model to the specified path.
    """
    # --- 1. Data Loading & Merging ---
    print(f"Loading datasets from {data_directory}...")
    try:
        orders = pd.read_parquet(f'{data_directory}/olist_orders_cleaned_dataset.parquet')
        items = pd.read_parquet(f'{data_directory}/olist_order_items_cleaned_dataset.parquet')
        customers = pd.read_parquet(f'{data_directory}/olist_customers_cleaned_dataset.parquet')
        sellers = pd.read_parquet(f'{data_directory}/olist_sellers_cleaned_dataset.parquet')
        products = pd.read_parquet(f'{data_directory}/olist_products_cleaned_dataset.parquet')
        geo = pd.read_parquet(f'{data_directory}/olist_geolocation_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        print(f"Error: A file was not found.\nDetails: {e}")
        return

    print("Merging datasets...")
    geo_agg = geo.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    df = orders.merge(items, on='order_id').merge(products, on='product_id').merge(sellers, on='seller_id').merge(customers, on='customer_id')
    df = df.merge(geo_agg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left').rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}).drop('geolocation_zip_code_prefix', axis=1)
    df = df.merge(geo_agg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left').rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}).drop('geolocation_zip_code_prefix', axis=1)

    # --- 2. Feature Engineering & Cleaning ---
    print("Engineering features...")
    time_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])

    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
    df = df.dropna(subset=['delivery_time_days'])
    
    original_rows = len(df)
    df = df[df['delivery_time_days'] <= 60]
    print(f"Filtered out {original_rows - len(df)} outlier deliveries (> 60 days).")
    df = df[df['delivery_time_days'] > 0]

    df = df.dropna(subset=['customer_lat', 'seller_lat'])
    df['distance_km'] = haversine_distance(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])
    df['product_volume_cm3'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df['processing_time_days'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / (24 * 3600)
    df['estimated_delivery_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (24 * 3600)
    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['distance_x_weight'] = df['distance_km'] * df['product_weight_g']
    
    df['customer_state'] = df['customer_state'].astype('category')
    df['seller_state'] = df['seller_state'].astype('category')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if df[col].dtype.name == 'category':
            if 'Unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('Unknown')
            df[col].fillna('Unknown', inplace=True)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(0, inplace=True)
            
    df = df[df['processing_time_days'] >= 0]

    # --- 3. Model Preparation & Training ---
    print("Preparing data and training model...")
    features = [
        'distance_km', 'processing_time_days', 'estimated_delivery_days',
        'price', 'freight_value', 'product_weight_g', 'product_volume_cm3',
        'purchase_day_of_week', 'purchase_hour', 'customer_state', 'seller_state',
        'distance_x_weight', 'customer_lat', 'customer_lng'
    ]
    target = 'delivery_time_days'
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    seller_avg_processing = X_train.groupby('seller_id')['processing_time_days'].mean().to_dict()
    global_processing_mean = X_train['processing_time_days'].mean()
    X_train['seller_avg_processing_time'] = X_train['seller_id'].map(seller_avg_processing).fillna(global_processing_mean)
    X_test['seller_avg_processing_time'] = X_test['seller_id'].map(seller_avg_processing).fillna(global_processing_mean)
    
    X_train['freight_x_seller_speed'] = X_train['freight_value'] * X_train['seller_avg_processing_time']
    X_test['freight_x_seller_speed'] = X_test['freight_value'] * X_test['seller_avg_processing_time']
    
    features.extend(['seller_avg_processing_time', 'freight_x_seller_speed'])
    X_train = X_train[features]
    X_test = X_test[features]

    param_grid = {
        'n_estimators': [400],
        'learning_rate': [0.1],
        'num_leaves': [60],
    }
    lgb_reg = lgb.LGBMRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=lgb_reg, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    
    # --- 4. Evaluation & Reporting ---
    print("\n" + "="*50)
    print("--- FINAL MODEL PERFORMANCE & ANALYSIS ---")
    print("="*50)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"R-squared (R²): {r2:.2%}\n")
    
    feature_importance_df = pd.DataFrame({
        'feature': best_model.feature_name_,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("--- MOST IMPORTANT FEATURES ---")
    print(feature_importance_df.to_string(index=False))

    # --- 5. Save the Final Model ---
    print("\n" + "="*50)
    print("--- SAVING FINAL MODEL ---")
    print("="*50)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    joblib.dump(best_model, model_save_path)
    print(f"Model saved successfully to:\n{model_save_path}")
    print("="*50)


if __name__ == '__main__':
    # Define the path where your data is located
    DATA_DIRECTORY = 'data/processed/files'
    
    # Define the exact path and filename for the saved model
    # Using a raw string (r'...') is important to handle backslashes correctly in Windows paths
    MODEL_SAVE_DIRECTORY = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Delivery Prediction'
    MODEL_FILENAME = 'delivery_time_predictor.joblib'
    full_model_path = os.path.join(MODEL_SAVE_DIRECTORY, MODEL_FILENAME)
    
    run_delivery_prediction_pipeline(DATA_DIRECTORY, full_model_path)