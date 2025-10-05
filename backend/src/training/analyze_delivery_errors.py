# backend/src/training/analyze_delivery_errors.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# --- Copy the haversine_distance function from your training script ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def analyze_model_errors():
    """
    Loads the trained delivery model and analyzes its prediction errors
    on the test set to find patterns.
    """
    print("--- Starting Delivery Model Error Analysis ---")

    # --- 1. Define Paths and Load Model ---
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_PATH = PROJECT_ROOT / "models"
    
    print("Step 1: Loading the saved delivery model...")
    try:
        model_path = MODELS_PATH / 'delivery_model_v2.joblib'
        pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please run the training script first.")
        return

    # --- 2. Recreate the EXACT same dataset and test set ---
    # This block is copied from the training script to ensure data consistency
    print("Step 2: Recreating the test set...")
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    try:
        # Load all necessary datasets
        orders = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_orders_cleaned_dataset.parquet')
        order_items = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_items_cleaned_dataset.parquet')
        products = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_products_cleaned_dataset.parquet')
        customers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_customers_cleaned_dataset.parquet')
        sellers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_sellers_cleaned_dataset.parquet')
        geolocation = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_geolocation_cleaned_dataset.parquet')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # All feature engineering steps must be identical to the training script
    df = orders.merge(order_items, on='order_id').merge(products, on='product_id').merge(customers, on='customer_id').merge(sellers, on='seller_id')
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
    df['is_same_state'] = (df['customer_state'] == df['seller_state']).astype(int)
    geo_avg = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    df = df.merge(geo_avg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}, inplace=True)
    df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
    df = df.merge(geo_avg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
    df.drop('geolocation_zip_code_prefix', axis=1, inplace=True)
    df['distance_km'] = haversine_distance(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])
    
    features = [
        'distance_km', 'freight_value', 'product_weight_g', 'product_volume_cm3',
        'seller_avg_shipping_time', 'seller_std_shipping_time', 'purchase_day_of_week',
        'purchase_month', 'is_same_state', 'seller_state', 'customer_state'
    ]

    numerical_features = [
    'distance_km', 'freight_value', 'product_weight_g', 'product_volume_cm3',
    'seller_avg_shipping_time', 'seller_std_shipping_time', 'purchase_day_of_week',
    'purchase_month', 'is_same_state'
    ]
    target = 'delivery_duration_days'
    
    final_df = df[features + [target]].dropna()
    final_df = final_df[final_df[target] > 0]
    X = final_df[features]
    y = final_df[target]
    
    # Use the same random_state to get the same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Make Predictions and Calculate Errors ---
    print("Step 3: Calculating prediction errors...")
    y_pred = pipeline.predict(X_test)
    
    # Create an analysis DataFrame
    analysis_df = X_test.copy()
    analysis_df['actual_duration'] = y_test
    analysis_df['predicted_duration'] = y_pred
    analysis_df['error'] = analysis_df['actual_duration'] - analysis_df['predicted_duration']
    analysis_df['absolute_error'] = np.abs(analysis_df['error'])
    
    # --- 4. Identify and Analyze Worst Predictions ---
    print("Step 4: Analyzing the worst prediction errors...")
    
    # Get the top 5% of worst predictions
    n_top_errors = int(len(analysis_df) * 0.05)
    worst_errors_df = analysis_df.sort_values('absolute_error', ascending=False).head(n_top_errors)
    
    print("\n--- ANALYSIS OF WORST ERRORS vs. ALL ERRORS ---")
    print("\n--- Feature Statistics ---")
    print("Comparing the average feature values:")
    print(pd.concat([
        analysis_df[numerical_features].mean().rename('Overall Avg'),
        worst_errors_df[numerical_features].mean().rename('Worst Errors Avg')
    ], axis=1))
    
    print("\n--- Customer State Distribution (Top 5) ---")
    print("Comparing the distribution of customer states:")
    print(pd.concat([
        (analysis_df['customer_state'].value_counts(normalize=True).head().rename('Overall %') * 100).round(1),
        (worst_errors_df['customer_state'].value_counts(normalize=True).head().rename('Worst Errors %') * 100).round(1)
    ], axis=1))

    # --- 5. Visualize the Errors ---
    print("\nStep 5: Generating visualizations...")
    
    # Plot 1: Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='actual_duration', y='predicted_duration', data=analysis_df, alpha=0.3, label='All Predictions')
    sns.scatterplot(x='actual_duration', y='predicted_duration', data=worst_errors_df, color='red', alpha=0.5, label=f'Top 5% Worst Errors')
    plt.plot([0, 150], [0, 150], 'k--', label='Perfect Prediction')
    plt.title('Actual vs. Predicted Delivery Duration')
    plt.xlabel('Actual Duration (days)')
    plt.ylabel('Predicted Duration (days)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(analysis_df['error'], kde=True, bins=50)
    plt.title('Distribution of Prediction Errors (Actual - Predicted)')
    plt.xlabel('Error in Days')
    plt.axvline(0, color='red', linestyle='--')
    plt.grid(True)
    plt.show()

    print("\n--- Error Analysis Finished ---")


if __name__ == '__main__':
    analyze_model_errors()