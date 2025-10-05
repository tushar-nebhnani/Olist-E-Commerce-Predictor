import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
from pathlib import Path # Use pathlib for robust path management

def train_satisfaction_model():
    """
    Loads cleaned data, engineers features, trains a satisfaction model,
    evaluates its performance, and saves the final pipeline.
    """
    print("--- Starting Satisfaction Model Training ---")

    # --- 1. Define Robust Paths ---
    # This makes the script runnable from any directory
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODELS_PATH = PROJECT_ROOT / "models"
    
    # --- 2. Load and Prepare Data ---
    print("Step 1 & 2: Loading, merging, and engineering features...")
    try:
        orders = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_orders_cleaned_dataset.parquet')
        reviews = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_reviews_cleaned_dataset.parquet')
        order_items = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_items_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure cleaned files are in '{PROCESSED_DATA_PATH}'.")
        return

    df = orders.merge(reviews, on='order_id').merge(order_items, on='order_id')
    
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['estimated_vs_actual_delivery'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days
    df['is_satisfied'] = (df['review_score'] >= 4).astype(int)

    # --- 3. Final Data Preparation ---
    features = ['price', 'freight_value', 'delivery_time_days', 'estimated_vs_actual_delivery']
    target = 'is_satisfied'
    
    final_df = df[features + [target]].dropna()
    X = final_df[features]
    y = final_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data preparation complete.")

    # --- 4. Create and Train the Model Pipeline ---
    print("Step 3: Building and training the model pipeline with SMOTE...")
    model_pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. Evaluate the Model (CRITICAL STEP) ---
    print("\nStep 4: Evaluating model performance on the test set...")
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Satisfied (0)', 'Satisfied (1)'])
    
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    # --- 6. Save the Final Pipeline ---
    print("Step 5: Saving the trained pipeline...")
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PATH / 'satisfaction_model_v1.joblib'
    joblib.dump(model_pipeline, model_path)
    print(f"✅ Pipeline saved successfully to {model_path}")
    
    print("\n--- Satisfaction Model Training Finished ---")

if __name__ == '__main__':
    train_satisfaction_model()