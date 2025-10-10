import pandas as pd
import numpy as np
import joblib
import os
import time # To track tuning time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# --- FILE PATHS ---
MASTER_DATA_FILE = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\processed\olist_price_recommender_master.csv'
MODEL_DIR = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Price Recommender'
MODEL_NAME_V2 = 'random_forest_price_recommender_v2.pkl'
MODEL_PATH_V2 = os.path.join(MODEL_DIR, MODEL_NAME_V2)

# --- 1. Load Data ---
try:
    df_olist = pd.read_csv(MASTER_DATA_FILE)
    print(f"Successfully loaded master dataset: {len(df_olist):,} rows.")
except FileNotFoundError:
    print(f"ERROR: Master data file not found at '{MASTER_DATA_FILE}'. Please check the path.")
    exit()

# --- 2. Data Preparation ---
TARGET = 'price'
ALL_FEATURES = [
    'product_category_name_english', 'freight_value', 'product_weight_g', 
    'product_length_cm', 'product_height_cm', 'product_width_cm', 'review_score' 
]
available_features = [f for f in ALL_FEATURES if f in df_olist.columns]

X = df_olist[available_features]
y = df_olist[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Preprocessing Setup (Base Pipeline) ---
numerical_features = [col for col in ['freight_value', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'review_score'] if col in available_features]
categorical_features = [col for col in ['product_category_name_english'] if col in available_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' 
)

# The base pipeline used for tuning (the regressor step will be tuned)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

# --- 4. Define Hyperparameter Search Space ---
# We define a dictionary of parameters to search. Note the 'regressor__' prefix.
param_dist = {
    # Number of trees in the forest
    'regressor__n_estimators': [100, 200, 300, 500],
    # Maximum number of levels in tree
    'regressor__max_depth': [10, 15, 20, 30, None],
    # Minimum number of samples required to split a node
    'regressor__min_samples_split': [2, 5, 10],
    # Minimum number of samples required at each leaf node
    'regressor__min_samples_leaf': [1, 2, 4],
    # Number of features to consider at every split
    'regressor__max_features': [0.6, 0.8, 1.0, 'sqrt'] # 0.6 = 60% of features
}

# --- 5. Hyperparameter Tuning (Randomized Search) ---
print("\n--- Starting Randomized Search Cross-Validation (Hyperparameter Tuning) ---")
start_time = time.time()

# Randomized search finds the best parameters by sampling from the defined distribution
# n_iter=50 means it will test 50 different parameter combinations
# cv=3 means 3-fold cross-validation
random_search = RandomizedSearchCV(
    pipeline, 
    param_distributions=param_dist, 
    n_iter=50, # Number of parameter settings that are sampled
    cv=3, 
    scoring='neg_mean_squared_error', # We optimize for the lowest MSE (highest negative MSE)
    verbose=2, 
    random_state=42, 
    n_jobs=-1
)

random_search.fit(X_train, y_train)
end_time = time.time()

print(f"\n[SUCCESS] Tuning Complete in {time.strftime('%M:%S', time.gmtime(end_time - start_time))}.")
print("----------------------------------------------------------------------")
print("Best Parameters Found:")
# The key 'regressor__...' is used because the parameter belongs to the 'regressor' step in the Pipeline
best_params = {k.replace('regressor__', ''): v for k, v in random_search.best_params_.items()}
print(best_params)

# --- 6. Final Evaluation of the Best Model ---
best_model = random_search.best_estimator_

y_pred_tuned = best_model.predict(X_test)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"\n--- Tuned Model Performance on Test Set ---")
print(f"Root Mean Squared Error (RMSE): R$ {rmse_tuned:,.2f}")
print(f"R-squared (R²): {r2_tuned:.4f}")

# --- 7. Save the Tuned Model ---
os.makedirs(MODEL_DIR, exist_ok=True) 
joblib.dump(best_model, MODEL_PATH_V2)
print(f"\n[SUCCESS] Tuned Model V2 saved to: '{MODEL_PATH_V2}'")

# --- 8. Example Recommendation Test with Tuned Model ---
loaded_model_v2 = joblib.load(MODEL_PATH_V2)

example_input = {
    'product_category_name_english': 'watches_gifts',
    'freight_value': 12.00,
    'product_weight_g': 350,
    'product_length_cm': 15,
    'product_height_cm': 5,
    'product_width_cm': 10,
    'review_score': 4.8 
}

# Reuse the recommend_price function logic
new_data = pd.DataFrame([example_input])
recommended_price_tuned = loaded_model_v2.predict(new_data)[0]

print("\n--- Example Price Recommendation (Tuned Model) ---")
print(f"💰 Recommended Selling Price (V2): R$ {round(recommended_price_tuned, 2):,.2f}")