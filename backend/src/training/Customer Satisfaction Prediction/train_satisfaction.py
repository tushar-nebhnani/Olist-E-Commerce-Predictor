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

    # --- 1. Define Robust Paths (MLOps Foundation) ---
    # This makes the script runnable from any directory by finding the project root.
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODELS_PATH = PROJECT_ROOT / "models"
    
    # Ensure the models directory exists for saving output files (Auditability)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    # --- 2. Load and Prepare Data ---
    print("Step 1 & 2: Loading, merging, and engineering features...")
    try:
        # Load pre-cleaned, essential e-commerce datasets
        orders = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, 'olist_orders_cleaned_dataset.parquet'))
        reviews = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, 'olist_order_reviews_cleaned_dataset.parquet'))
        order_items = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, 'olist_order_items_cleaned_dataset.parquet'))
    except FileNotFoundError as e:
        # Handle missing data files gracefully
        print(f"Error: {e}. Ensure cleaned files are in '{PROCESSED_DATA_PATH}'.")
        return

    # Merge datasets on the common key 'order_id'
    df = orders.merge(reviews, on='order_id').merge(order_items, on='order_id')
    
    # Convert necessary columns to datetime objects for time-based calculations
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- Feature Engineering: Creating Predictive Signals ---
    # 1. Calculate the actual delivery duration
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    # 2. Calculate the difference between estimated and actual delivery (Performance vs. Expectation)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data preparation complete.")

    # --- 4. SMOTE ANALYSIS & AUDIT: Isolating Real and Synthetic Data ---
    
    # 4a. Isolate the real minority class (Not Satisfied) in the training set for inspection
    not_satisfied_real_df = X_train[y_train == 0].copy()
    
    print(f"\nStep 4a: Real Not Satisfied Orders (Minority Class Size): {len(not_satisfied_real_df)}")
    print("Sample of Real Not Satisfied Data (Focus on Risk Signatures):\n")
    print(not_satisfied_real_df[['estimated_vs_actual_delivery', 'delivery_time_days', 'price']].head())
    
    # Save the real minority data for auditability and defense
    real_audit_path = os.path.join(MODELS_PATH, 'real_not_satisfied.csv')
    not_satisfied_real_df.to_csv(real_audit_path, index=False)
    print(f"✅ Real minority data saved to {real_audit_path}")


    # 4b. Explicitly run SMOTE on the training data to generate synthetic samples
    smote_sampler = SMOTE(random_state=42)
    # NOTE: This step is purely for calculating and inspecting the synthetic data before the pipeline fit.
    X_smote, y_smote = smote_sampler.fit_resample(X_train, y_train)

    # Calculate the number of synthetic samples created
    num_synthetic = len(X_smote) - len(X_train)
    
    # The synthetic data is appended to the end of the X_smote DataFrame.
    synthetic_df = X_smote.tail(num_synthetic)
    
    print(f"\nStep 4b: Synthetic Not Satisfied Orders Created: {num_synthetic}")
    print(f"Total rows after SMOTE (Balanced 1:1): {len(X_smote)}")
    print("\nSample of Synthetic Data (Must be Plausible & Non-identical):\n")
    print(synthetic_df[['estimated_vs_actual_delivery', 'delivery_time_days', 'price']].head())
    
    # Save the synthetic data for auditability and defense
    synthetic_audit_path = os.path.join(MODELS_PATH, 'synthetic_not_satisfied.csv')
    synthetic_df.to_csv(synthetic_audit_path, index=False)
    print(f"✅ Synthetic data saved to {synthetic_audit_path}")

    
    # --- 5. Create and Train the Model Pipeline (The Final Production Engine) ---
    print("\nStep 5: Building and training the final production pipeline...")
    # ImbPipeline is used to ensure SMOTE is correctly applied only during training, preventing data leakage.
    model_pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])
    
    # Train the entire pipeline (SMOTE + LGBM) on the original training data
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # --- 6. Evaluate the Model (CRITICAL STEP) ---
    print("\nStep 6: Evaluating model performance on the held-out test set...")
    y_pred = model_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Satisfied (0)', 'Satisfied (1)'])
    
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)

    # --- 7. Save the Final Pipeline (MLOps Readiness) ---
    print("Step 7: Saving the trained pipeline...")
    model_path = MODELS_PATH / 'satisfaction_model_v1.joblib'
    # Save the entire pipeline object (SMOTE logic + trained LGBM) using joblib
    joblib.dump(model_pipeline, model_path)
    print(f"✅ Pipeline saved successfully to {model_path}")
    
    print("\n--- Satisfaction Model Training Finished ---")

if __name__ == '__main__':
    train_satisfaction_model()
