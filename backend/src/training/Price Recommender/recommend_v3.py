import pandas as pd
import numpy as np
import joblib
import os
import time
from xgboost import XGBRegressor # Use XGBoost
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# --- FILE PATHS (Using the NEW V3 MASTER FILE) ---
DATA_PATH = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\processed'
MASTER_DATA_FILE_V3 = os.path.join(DATA_PATH, 'olist_price_recommender_master_v3.csv')
MODEL_DIR = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Price Recommender'
MODEL_NAME_V3 = 'xgboost_price_recommender_v3.pkl' # Renamed for new algorithm and features
MODEL_PATH_V3 = os.path.join(MODEL_DIR, MODEL_NAME_V3)

# --- 1. Load V3 Master Data ---
try:
    df_olist = pd.read_csv(MASTER_DATA_FILE_V3)
    print(f"Successfully loaded V3 master dataset: {len(df_olist):,} rows.")
except FileNotFoundError:
    print(f"ERROR: V3 Master data file not found at '{MASTER_DATA_FILE_V3}'. Please run Step 2A first.")
    exit()

# --- 2. Data Preparation ---
TARGET = 'price'
# Updated feature list to include the newly engineered features
NEW_FEATURES = [
    'product_category_name_english', 'freight_value', 'product_weight_g', 
    'product_length_cm', 'product_height_cm', 'product_width_cm', 'review_score',
    'seller_avg_review_score', 'product_competition_count', 'price_per_volume' 
]
available_features = [f for f in NEW_FEATURES if f in df_olist.columns]

X = df_olist[available_features]
y = df_olist[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Preprocessing Setup ---
# All new features except category translation are numerical
numerical_features = [f for f in available_features if f not in ['product_category_name_english']]
categorical_features = ['product_category_name_english']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' 
)

# Base pipeline using XGBRegressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # Using XGBRegressor for better performance
    ('regressor', XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'))
])

# --- 4. Define XGBoost Hyperparameter Search Space ---
param_dist_xgb = {
    'regressor__n_estimators': [200, 500, 800],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_depth': [5, 8, 12],
    'regressor__min_child_weight': [1, 5, 10],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0],
}

# --- 5. Hyperparameter Tuning (Randomized Search) ---
print("\n--- Starting XGBoost Randomized Search Cross-Validation (Model V3) ---")
start_time = time.time()

random_search_xgb = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_dist_xgb, 
    n_iter=30, # Reduced iterations due to XGBoost's complexity and time
    cv=3, 
    scoring='neg_mean_squared_error',
    verbose=1, 
    random_state=42, 
    n_jobs=-1
)

random_search_xgb.fit(X_train, y_train)
end_time = time.time()

print(f"\n[SUCCESS] Tuning Complete in {time.strftime('%M:%S', time.gmtime(end_time - start_time))}.")
print("----------------------------------------------------------------------")
print("Best XGBoost Parameters Found:")
best_params_xgb = {k.replace('regressor__', ''): v for k, v in random_search_xgb.best_params_.items()}
print(best_params_xgb)

# --- 6. Final Evaluation of the Best Model (V3) ---
best_model_v3 = random_search_xgb.best_estimator_

y_pred_v3 = best_model_v3.predict(X_test)
mse_v3 = mean_squared_error(y_test, y_pred_v3)
rmse_v3 = np.sqrt(mse_v3)
r2_v3 = r2_score(y_test, y_pred_v3)

print(f"\n--- XGBoost Model Performance (V3) ---")
print(f"Root Mean Squared Error (RMSE): R$ {rmse_v3:,.2f}")
print(f"R-squared (R²): {r2_v3:.4f}")

# --- 7. Save the Tuned Model (V3) ---
os.makedirs(MODEL_DIR, exist_ok=True) 
joblib.dump(best_model_v3, MODEL_PATH_V3)
print(f"\n[SUCCESS] XGBoost Model V3 saved to: '{MODEL_PATH_V3}'")