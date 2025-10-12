import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
import joblib  # Used for saving and loading Python objects, essential for model persistence
from sklearn.decomposition import TruncatedSVD # The core model for Matrix Factorization

# --- Configuration Section ---
# Define the paths for data input and model output artifacts.
# NOTE: These paths are specific to the user's environment and should be parameterized in a production setup.
DATA_PATH = r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\processed"
ARTIFACTS_PATH = r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Product Recommendation"

def train_and_save_svd_model():
    """
    Implements a Collaborative Filtering approach using Matrix Factorization 
    (specifically, Truncated Singular Value Decomposition).

    The function loads raw user-item interaction data, transforms it into a 
    sparse user-item matrix, trains the SVD model, and saves the trained 
    model along with the mapping dictionaries (artifacts) required for 
    real-time inference.
    """
    print("--- 1. Loading and Preparing Data ---")
    try:
        # Load cleaned order and order item data, typically containing 
        # customer_id, product_id, and timestamps.
        orders_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_orders_cleaned_dataset.parquet'))
        order_items_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_order_items_cleaned_dataset.parquet'))
        print("✅ Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files. Details: {e}")
        return

    # Merge to link customers (users) with the products (items) they bought.
    df = pd.merge(orders_df, order_items_df, on='order_id')
    
    # Aggregate interactions to create the fundamental user-item interaction dataset.
    # The size() function implicitly counts interactions (purchases in this case).
    interactions_df = df.groupby(['customer_id', 'product_id']).size().reset_index(name='purchase_count')
    
    # Binarization: Convert the 'purchase_count' to 1. This treats the problem 
    # as Implicit Feedback, where the model learns from 'purchase/no-purchase' 
    # rather than a quantitative rating.
    interactions_df['purchase_count'] = 1
    print(f"✅ Data prepared with {len(interactions_df)} unique user-item interactions.")

    print("\n--- 2. Building User-Item Matrix ---")
    
    # Convert string IDs to categorical types. This is critical for generating 
    # contiguous, zero-based integer codes required for the sparse matrix.
    user_cat = interactions_df['customer_id'].astype('category')
    product_cat = interactions_df['product_id'].astype('category')
    
    # Get integer codes (0 to N-1) for customers and products.
    row_indices = user_cat.cat.codes
    col_indices = product_cat.cat.codes
    
    # Construct the User-Item Interaction Matrix as a Compressed Sparse Row (CSR) matrix.
    # This is highly memory-efficient for sparse data (many zero entries, i.e., 
    # customers who haven't bought most products).
    user_item_sparse_matrix = csr_matrix((interactions_df['purchase_count'], (row_indices, col_indices)), 
                                         shape=(len(user_cat.cat.categories), len(product_cat.cat.categories)))
    
    # Create mapping dictionaries (artifacts) to convert between the original 
    # string IDs and the integer indices used by the model. These are vital for prediction.
    user_map = {i: user_id for i, user_id in enumerate(user_cat.cat.categories)}
    product_map = {i: product_id for i, product_id in enumerate(product_cat.cat.categories)}
    print("✅ User-Item sparse matrix created efficiently.")

    # --- Model Training with TruncatedSVD ---
    print("\n--- 3. Training The Engine with TruncatedSVD ---")
    # Initialize TruncatedSVD, a linear dimensionality reduction technique suitable 
    # for sparse data and often used as a baseline for Matrix Factorization in CF.
    # n_components=100 defines the dimensionality of the latent factor space.
    svd = TruncatedSVD(n_components=100, random_state=42)
    print("⏳ Fitting the SVD model...")
    
    # Fit the model: Decompose the User-Item matrix (R) into two lower-rank matrices:
    # U (User Latent Factors) and V^T (Item Latent Factors).
    # R ≈ U * V^T, where U has shape (num_users, n_components) and V^T has 
    # shape (n_components, num_items).
    svd.fit(user_item_sparse_matrix)
    print("✅ Engine training complete!")

    print("\n--- 4. Generating Sample Recommendations to Assess Quality ---")
    
    # Reconstruct the original matrix (Predicted Ratings/Scores) by multiplying the 
    # transformed (User Latent Factors) with the transposed components (Item Latent Factors).
    # predicted_ratings = U * V^T
    predicted_ratings = np.dot(svd.transform(user_item_sparse_matrix), svd.components_)
    
    # Select a random user for a simple, qualitative evaluation.
    random_user_index = np.random.choice(predicted_ratings.shape[0])
    random_user_id = user_map[random_user_index]
    
    # Get the predicted scores for all products for the selected user.
    user_ratings = predicted_ratings[random_user_index]
    all_product_ids = list(product_map.values())
    
    # Combine product IDs with predicted scores into a DataFrame.
    recommendations = pd.DataFrame({'product_id': all_product_ids, 'score': user_ratings})
    
    # Filter out items the user has already bought to ensure novel recommendations 
    # (a common practice in recommender systems).
    bought_items = interactions_df[interactions_df['customer_id'] == random_user_id]['product_id']
    final_recommendations = recommendations[~recommendations['product_id'].isin(bought_items)]
    
    # Sort by score and take the top N recommendations.
    final_recommendations = final_recommendations.sort_values(by='score', ascending=False).head(10)
    print(f"\nTop 10 recommendations for user '{random_user_id}':")
    print(final_recommendations)

    print("\n--- 5. Saving Model and Artifacts to Disk ---")
    
    # Ensure the artifact directory exists before saving.
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    
    # Save artifacts using joblib for fast and reliable serialization.
    # The SVD model, product_map, and user_map are all necessary to load 
    # the model later and perform predictions on new data.
    joblib.dump(svd, os.path.join(ARTIFACTS_PATH, 'svd_model.joblib'))
    joblib.dump(product_map, os.path.join(ARTIFACTS_PATH, 'svd_product_map.joblib'))
    joblib.dump(user_map, os.path.join(ARTIFACTS_PATH, 'svd_user_map.joblib'))
    
    print("✅ All artifacts saved successfully using joblib!")
    print(f"   - Saved to: {os.path.join(ARTIFACTS_PATH, 'svd_model.joblib')}")
    print(f"   - Saved to: {os.path.join(ARTIFACTS_PATH, 'svd_product_map.joblib')}")
    print(f"   - Saved to: {os.path.join(ARTIFACTS_PATH, 'svd_user_map.joblib')}")

if __name__ == "__main__":
    train_and_save_svd_model()