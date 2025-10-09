import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import joblib  # Switched from pickle to joblib
from sklearn.decomposition import TruncatedSVD

# --- Configuration ---
DATA_PATH = r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\processed"
# Updated path to save the final model artifacts
ARTIFACTS_PATH = r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Product Recommendation"

def train_and_save_svd_model():
    """
    Loads data, trains a Matrix Factorization model using TruncatedSVD,
    and saves the model and mappings using joblib.
    """
    print("--- 1. Loading and Preparing Data ---")
    try:
        orders_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_orders_cleaned_dataset.parquet'))
        order_items_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_order_items_cleaned_dataset.parquet'))
        print("✅ Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files. Details: {e}")
        return

    # --- Data Preparation ---
    df = pd.merge(orders_df, order_items_df, on='order_id')
    interactions_df = df.groupby(['customer_id', 'product_id']).size().reset_index(name='purchase_count')
    interactions_df['purchase_count'] = 1
    print(f"✅ Data prepared with {len(interactions_df)} unique user-item interactions.")

    # --- Create User-Item Matrix (Memory-Efficiently) ---
    print("\n--- 2. Building User-Item Matrix ---")
    user_cat = interactions_df['customer_id'].astype('category')
    product_cat = interactions_df['product_id'].astype('category')
    row_indices = user_cat.cat.codes
    col_indices = product_cat.cat.codes
    user_item_sparse_matrix = csr_matrix((interactions_df['purchase_count'], (row_indices, col_indices)), 
                                         shape=(len(user_cat.cat.categories), len(product_cat.cat.categories)))
    user_map = {i: user_id for i, user_id in enumerate(user_cat.cat.categories)}
    product_map = {i: product_id for i, product_id in enumerate(product_cat.cat.categories)}
    print("✅ User-Item sparse matrix created efficiently.")

    # --- Model Training with TruncatedSVD ---
    print("\n--- 3. Training The Engine with TruncatedSVD ---")
    svd = TruncatedSVD(n_components=100, random_state=42)
    print("⏳ Fitting the SVD model...")
    svd.fit(user_item_sparse_matrix)
    print("✅ Engine training complete!")

    # --- Performance Evaluation (Qualitative) ---
    print("\n--- 4. Generating Sample Recommendations to Assess Quality ---")
    predicted_ratings = np.dot(svd.transform(user_item_sparse_matrix), svd.components_)
    random_user_index = np.random.choice(predicted_ratings.shape[0])
    random_user_id = user_map[random_user_index]
    user_ratings = predicted_ratings[random_user_index]
    all_product_ids = list(product_map.values())
    recommendations = pd.DataFrame({'product_id': all_product_ids, 'score': user_ratings})
    bought_items = interactions_df[interactions_df['customer_id'] == random_user_id]['product_id']
    final_recommendations = recommendations[~recommendations['product_id'].isin(bought_items)]
    final_recommendations = final_recommendations.sort_values(by='score', ascending=False).head(10)
    print(f"\nTop 10 recommendations for user '{random_user_id}':")
    print(final_recommendations)

    # --- Save the Model and Mappings using Joblib ---
    print("\n--- 5. Saving Model and Artifacts to Disk ---")
    
    # Ensure the target directory exists
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    
    # Save artifacts using joblib
    joblib.dump(svd, os.path.join(ARTIFACTS_PATH, 'svd_model.joblib'))
    joblib.dump(product_map, os.path.join(ARTIFACTS_PATH, 'svd_product_map.joblib'))
    joblib.dump(user_map, os.path.join(ARTIFACTS_PATH, 'svd_user_map.joblib'))
    
    print("✅ All artifacts saved successfully using joblib!")
    print(f"   - Saved to: {os.path.join(ARTIFACTS_PATH, 'svd_model.joblib')}")
    print(f"   - Saved to: {os.path.join(ARTIFACTS_PATH, 'svd_product_map.joblib')}")
    print(f"   - Saved to: {os.path.join(ARTIFACTS_PATH, 'svd_user_map.joblib')}")

if __name__ == "__main__":
    train_and_save_svd_model()
