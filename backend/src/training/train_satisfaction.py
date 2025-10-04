# backend/src/training/train_satisfaction.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_satisfaction_model_from_clean_files():
    """
    This script loads individual cleaned datasets, merges them, engineers final 
    features, trains a satisfaction model using SMOTE, and saves the final pipeline.
    """
    print("--- Starting Satisfaction Model Training from Clean Files ---")

    # --- 1. Load Pre-Cleaned Data ---
    print("Step 1: Loading individual clean datasets...")
    PROCESSED_DATA_PATH = os.path.join("..", "..", "data", "processed")
    
    try:
        orders = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'orders_cleaned_dataset'))
        reviews = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'orders_review_cleaned_dataset'))
        order_items = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'order_items_cleaned_dataset'))
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the cleaned CSV files are in the 'backend/data/processed/' directory.")
        return

    # --- 2. Merge and Final Feature Engineering ---
    print("Step 2: Merging data and engineering final features...")
    df = orders.merge(reviews, on='order_id')
    df = df.merge(order_items, on='order_id')

    # Convert date columns to datetime objects
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Create time-based features
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['estimated_vs_actual_delivery'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days
    
    # Create the target variable
    df['is_satisfied'] = (df['review_score'] >= 4).astype(int)

    # --- 3. Final Data Preparation for Model ---
    features = ['price', 'freight_value', 'delivery_time_days', 'estimated_vs_actual_delivery']
    target = 'is_satisfied'
    
    final_df = df[features + [target]].dropna()
    X = final_df[features]
    y = final_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data preparation complete.")

    # --- 4. Create and Train the Model Pipeline ---
    print("Step 4: Building and training the model pipeline with SMOTE...")
    model_pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. Save the Final Pipeline ---
    print("Step 5: Saving the trained pipeline...")
    MODELS_PATH = os.path.join("..", "..", "models")
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
        
    joblib.dump(model_pipeline, os.path.join(MODELS_PATH, 'satisfaction_model_v1.joblib'))
    print(f"Pipeline saved successfully to {os.path.join(MODELS_PATH, 'satisfaction_model_v1.joblib')}")
    
    print("--- Satisfaction Model Training Finished ---")


if __name__ == '__main__':
    train_satisfaction_model_from_clean_files()