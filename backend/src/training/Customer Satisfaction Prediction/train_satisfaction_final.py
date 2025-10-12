import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline # CHANGED: Using the standard scikit-learn Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
from xgboost import XGBClassifier 

# Haversine distance function (no change)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def train_satisfaction_final_model():
    """
    Trains the final satisfaction model using the enhanced V2 feature set
    and an XGBoost classifier with class weighting.
    """
    print("--- Starting Final Model Training (XGBoost with Class Weighting) ---")

    # --- 1. Define Paths (No change) ---
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODELS_PATH = PROJECT_ROOT / "models"
    
    # --- 2. Load and Prepare Data (Same as enhanced V2) ---
    print("Step 1: Loading all necessary datasets...")
    try:
        orders = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_orders_cleaned_dataset.parquet')
        reviews = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_reviews_cleaned_dataset.parquet')
        order_items = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_items_cleaned_dataset.parquet')
        products = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_products_cleaned_dataset.parquet')
        payments = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_payments_cleaned_dataset.parquet')
        customers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_customers_cleaned_dataset.parquet')
        sellers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_sellers_cleaned_dataset.parquet')
        geolocation = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_geolocation_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure cleaned files are in '{PROCESSED_DATA_PATH}'.")
        return

    # Merge core datasets
    df = orders.merge(reviews, on='order_id').merge(order_items, on='order_id').merge(products, on='product_id').merge(payments, on='order_id').merge(customers, on='customer_id').merge(sellers, on='seller_id')

    # --- 3. Advanced Feature Engineering  ---
    print("Step 2: Engineering advanced features...")
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['delivery_time_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['estimated_vs_actual_delivery'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days
    df['is_satisfied'] = (df['review_score'] >= 4).astype(int)

    seller_stats = df.groupby('seller_id').agg(seller_avg_review_score=('review_score', 'mean'), seller_order_count=('order_id', 'nunique')).reset_index()
    df = df.merge(seller_stats, on='seller_id', how='left')

    geo_avg = geolocation.groupby('geolocation_zip_code_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()
    df = df.merge(geo_avg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}, inplace=True)
    df = df.drop('geolocation_zip_code_prefix', axis=1)
    df = df.merge(geo_avg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
    df = df.drop('geolocation_zip_code_prefix', axis=1)
    df['distance_km'] = haversine_distance(df['customer_lat'], df['customer_lng'], df['seller_lat'], df['seller_lng'])
    
    top_categories = df['product_category_name'].value_counts().nlargest(20).index
    df['product_category_name'] = df['product_category_name'].where(df['product_category_name'].isin(top_categories), 'Other')

    features = [
        'price', 'freight_value', 'delivery_time_days', 'estimated_vs_actual_delivery',
        'payment_installments', 'payment_value', 'product_photos_qty', 'product_weight_g',
        'product_category_name', 'seller_avg_review_score', 'seller_order_count', 'distance_km'
    ]
    target = 'is_satisfied'
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 4. Calculate Class Weight for XGBoost ---
    # NEW: This is the core of the new strategy. We calculate a weight to force the model
    # to pay more attention to the minority class ('Not Satisfied').
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

    # --- 5. Create Preprocessing and XGBoost Pipeline ---
    print("Step 3: Building preprocessing and XGBoost pipeline...")
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # CHANGED: The pipeline now uses XGBClassifier and does NOT use SMOTE.
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ])

    # --- 6. Hyperparameter Tuning & Training ---
    print("Step 4: Performing hyperparameter tuning for XGBoost...")
    # CHANGED: Parameter grid is now for XGBoost
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # --- 7. Evaluate and Save ---
    print("\nStep 5: Evaluating final model performance...")
    y_pred = best_model.predict(X_test)
    print("Final Model Classification Report:\n", classification_report(y_test, y_pred))

    print("Step 6: Saving the final trained pipeline...")
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_PATH / 'satisfaction_model_final.joblib'
    joblib.dump(best_model, model_path)
    print(f"✅ Final pipeline saved successfully to {model_path}")
    
    print("\n--- Final Model Training Finished ---")

if __name__ == '__main__':
    train_satisfaction_final_model()