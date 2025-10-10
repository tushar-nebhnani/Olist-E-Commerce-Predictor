import pandas as pd
import numpy as np
import joblib
import os # Import the os module for path manipulation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Define File Path and Load Data ---
# Ensure this path points to the CSV file created in the previous step.
MASTER_DATA_FILE = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\processed\olist_price_recommender_master.csv'

try:
    # Reading the data is confirmed successful by your previous output
    df_olist = pd.read_csv(MASTER_DATA_FILE)
    # The rest of the setup and training code remains the same...
# --- (Training code is identical to what you provided) ---
    print(f"Successfully loaded master dataset: {len(df_olist):,} rows.")
except FileNotFoundError:
    print(f"ERROR: Master data file not found at '{MASTER_DATA_FILE}'. Please check the path and file name.")
    exit()

# --- 2. Data Preparation for Modeling ---
TARGET = 'price'
ALL_FEATURES = [
    'product_category_name_english', 
    'freight_value', 
    'product_weight_g', 
    'product_length_cm', 
    'product_height_cm', 
    'product_width_cm', 
    'review_score' 
]
available_features = [f for f in ALL_FEATURES if f in df_olist.columns]
print(f"Using features: {available_features}")

X = df_olist[available_features]
y = df_olist[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Create Preprocessing and Model Pipeline ---
numerical_features = [col for col in ['freight_value', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'review_score'] if col in available_features]
categorical_features = [col for col in ['product_category_name_english'] if col in available_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='drop' 
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100, 
        max_depth=15,          
        random_state=42, 
        n_jobs=-1 
    ))
])

# --- 4. Train and Evaluate the Model ---
print("\nTraining the Random Forest Regressor... This may take a few minutes.")
model.fit(X_train, y_train)
print("[SUCCESS] Model training complete.")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Performance on Test Set ---")
print(f"Root Mean Squared Error (RMSE): R$ {rmse:,.2f}")
print(f"R-squared (R²): {r2:.4f}")

# --- 5. Save the Trained Model to the Specified Path ---

MODEL_DIR = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Price Recommender'
MODEL_NAME = 'random_forest_price_recommender_v1.pkl' # Descriptive name for the base model
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Ensure the directory exists before saving
os.makedirs(MODEL_DIR, exist_ok=True) 

joblib.dump(model, MODEL_PATH)
print(f"\n[SUCCESS] Model saved to: '{MODEL_PATH}'")

# --- 6. Recommendation Function (for API Integration) ---
def recommend_price(input_data: dict, model_pipeline: Pipeline) -> float:
    """Generates a price recommendation based on product features."""
    new_data = pd.DataFrame([input_data])
    predicted_price = model_pipeline.predict(new_data)[0]
    return round(predicted_price, 2)

# --- 7. Example Recommendation Test ---
# Load the model from the newly saved path to confirm saving/loading works
loaded_model = joblib.load(MODEL_PATH)

example_input = {
    'product_category_name_english': 'watches_gifts',
    'freight_value': 12.00,
    'product_weight_g': 350,
    'product_length_cm': 15,
    'product_height_cm': 5,
    'product_width_cm': 10,
    'review_score': 4.8 
}

recommended_price = recommend_price(example_input, loaded_model)

print("\n--- Example Price Recommendation for API Test ---")
print(f"Input Product: {example_input['product_category_name_english']} (Score: {example_input['review_score']})")
print(f"💰 Recommended Selling Price: R$ {recommended_price:,.2f}")