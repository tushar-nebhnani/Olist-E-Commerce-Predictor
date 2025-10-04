# backend/src/training/train_satisfaction_v2.py

import pandas as pd
import numpy as np # ADDED: For numpy types
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # ADDED: To handle missing values in the pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

def train_satisfaction_model_v2():
    """
    Trains an improved (V2) satisfaction model with more features and hyperparameter tuning.
    """
    print("--- Starting V2 Satisfaction Model Training ---")

    # --- 1. Define Robust Paths ---
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODELS_PATH = PROJECT_ROOT / "models"
    
    # --- 2. Load and Prepare Data ---
    print("Step 1: Loading all necessary clean datasets for V2...")
    try:
        orders = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_orders_cleaned_dataset.parquet')
        reviews = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_reviews_cleaned_dataset.parquet')
        order_items = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_items_cleaned_dataset.parquet')
        products = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_products_cleaned_dataset.parquet')
        payments = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_payments_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure cleaned files are in '{PROCESSED_DATA_PATH}'.")
        return

    # Merge all datasets
    df = orders.merge(reviews, on='order_id')
    df = df.merge(order_items, on='order_id')
    df = df.merge(products, on='product_id')
    df = df.merge(payments, on='order_id')
    
    # --- 3. Feature Engineering for V2 ---
    print("Step 2: Engineering a richer feature set...")
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['estimated_vs_actual_delivery'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days
    df['is_satisfied'] = (df['review_score'] >= 4).astype(int)

    # --- 4. Handle High-Cardinality Categorical Feature ---
    # ADDED: This block intelligently reduces the number of product categories
    print("Handling high-cardinality for 'product_category_name'...")
    top_categories = df['product_category_name'].value_counts().nlargest(20).index
    df['product_category_name'] = df['product_category_name'].where(df['product_category_name'].isin(top_categories), 'Other')

    # --- 5. Final Data Preparation ---
    features = [
        'price', 'freight_value', 'delivery_time_days', 'estimated_vs_actual_delivery',
        'payment_installments', 'payment_value', 'product_photos_qty', 'product_weight_g',
        'product_category_name'
    ]
    target = 'is_satisfied'
    
    # CHANGED: We removed the aggressive .dropna() to let the pipeline handle missing values
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 6. Create a Robust Preprocessing and Model Pipeline ---
    print("Step 3: Building V2 preprocessing and model pipeline...")
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # CHANGED: Added imputers to both pipelines to handle NaNs gracefully
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', lgb.LGBMClassifier(random_state=42))
    ])

    # --- 7. Hyperparameter Tuning & Training ---
    print("Step 4: Performing hyperparameter tuning...")
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [-1, 10]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # --- 8. Evaluate and Save ---
    print("\nStep 5: Evaluating best V2 model performance...")
    y_pred = best_model.predict(X_test)
    print("V2 Model Classification Report:\n", classification_report(y_test, y_pred))

    print("Step 6: Saving the V2 trained pipeline...")
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PATH / 'satisfaction_model_v2.joblib'
    joblib.dump(best_model, model_path)
    print(f"✅ V2 Pipeline saved successfully to {model_path}")
    
    print("\n--- V2 Satisfaction Model Training Finished ---")

if __name__ == '__main__':
    train_satisfaction_model_v2()