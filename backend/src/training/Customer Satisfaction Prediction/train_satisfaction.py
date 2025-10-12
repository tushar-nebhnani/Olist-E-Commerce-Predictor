import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# ImbPipeline is used to correctly sequence SMOTE (resampling) and the Classifier
from imblearn.pipeline import Pipeline as ImbPipeline
# SMOTE is used to handle class imbalance by over-sampling the minority class (Not Satisfied)
from imblearn.over_sampling import SMOTE
import joblib
import os
from pathlib import Path # Use pathlib for robust path management

def train_satisfaction_model():
    """
    Loads cleaned data, engineers features, trains a satisfaction model (binary classification),
    evaluates its performance, and saves the final pipeline for deployment.
    
    Target Variable: 'is_satisfied' (1 if review_score >= 4, 0 otherwise)
    Model: LightGBM Classifier within an imblearn Pipeline (with SMOTE).
    """
    print("--- Starting Satisfaction Model Training ---")

    # --- 1. Define Robust Paths ---
    # This makes the script runnable from any directory by finding the project root.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODELS_PATH = PROJECT_ROOT / "models"
    
    # --- 2. Load and Prepare Data ---
    print("Step 1 & 2: Loading, merging, and engineering features...")
    try:
        # Load pre-cleaned, essential e-commerce datasets
        orders = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_orders_cleaned_dataset.parquet')
        reviews = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_reviews_cleaned_dataset.parquet')
        order_items = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_items_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        # Handle missing data files gracefully
        print(f"Error: {e}. Ensure cleaned files are in '{PROCESSED_DATA_PATH}'.")
        return

    # Merge datasets on the common key 'order_id'
    df = orders.merge(reviews, on='order_id').merge(order_items, on='order_id')
    
    # Convert necessary columns to datetime objects for time-based calculations
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- Feature Engineering: Creating Predictive Signals from Timestamps ---
    # 1. Calculate the actual delivery duration
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    # 2. Calculate the difference between estimated and actual delivery (a measure of delivery performance/speed)
    # A positive number means delivery was earlier than estimated (a good sign for satisfaction)
    df['estimated_vs_actual_delivery'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days
    # 3. Define the Binary Target Variable: Customer Satisfaction (1=Satisfied, 0=Not Satisfied)
    df['is_satisfied'] = (df['review_score'] >= 4).astype(int)

    # --- 3. Final Data Preparation ---
    # Select the features (X) for the model
    features = ['price', 'freight_value', 'delivery_time_days', 'estimated_vs_actual_delivery']
    # Define the target variable (y)
    target = 'is_satisfied'
    
    # Drop rows with any missing values in the selected features or target
    final_df = df[features + [target]].dropna()
    X = final_df[features]
    y = final_df[target]

    # Split data into training (80%) and testing (20%) sets
    # stratify=y ensures the proportion of 'Satisfied' (1) and 'Not Satisfied' (0) is equal in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data preparation complete.")

    # --- 4. Create and Train the Model Pipeline ---
    print("Step 3: Building and training the model pipeline with SMOTE...")
    # Use ImbPipeline to apply SMOTE *only* to the training data before feeding it to the classifier.
    model_pipeline = ImbPipeline(steps=[
        # SMOTE: Over-samples the minority class (Not Satisfied) in the training data to balance the classes
        ('smote', SMOTE(random_state=42)),
        # Classifier: LightGBM (LGBM) is chosen for its speed and efficiency in classification tasks
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])
    
    # Train the entire pipeline (SMOTE + LGBM) on the training data
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. Evaluate the Model (CRITICAL STEP) ---
    print("\nStep 4: Evaluating model performance on the test set...")
    # Predict on the held-out test set (unseen data) for an unbiased evaluation
    y_pred = model_pipeline.predict(X_test)
    
    # Calculate overall prediction accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Generate a detailed report showing Precision, Recall, and F1-Score for each class
    report = classification_report(y_test, y_pred, target_names=['Not Satisfied (0)', 'Satisfied (1)'])
    
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    # --- 6. Save the Final Pipeline ---
    print("Step 5: Saving the trained pipeline...")
    # Create the models directory if it doesn't exist
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PATH / 'satisfaction_model_v1.joblib'
    # Save the entire pipeline object (SMOTE logic + trained LGBM) using joblib
    joblib.dump(model_pipeline, model_path)
    print(f"✅ Pipeline saved successfully to {model_path}")
    
    print("\n--- Satisfaction Model Training Finished ---")

if __name__ == '__main__':
    train_satisfaction_model()