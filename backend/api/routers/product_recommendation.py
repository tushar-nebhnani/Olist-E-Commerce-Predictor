from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import os
import joblib
import logging

# --- 1. Initialize Router and Set Up Paths ---
router = APIRouter()
logging.basicConfig(level=logging.INFO)

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    ARTIFACTS_PATH = PROJECT_ROOT / "models" / "Product Recommendation"
    DATA_PATH = PROJECT_ROOT / "data" / "processed"
except IndexError:
    PROJECT_ROOT = Path.cwd()
    ARTIFACTS_PATH = PROJECT_ROOT / "models" / "Product Recommendation"
    DATA_PATH = PROJECT_ROOT / "data" / "processed"

# --- 2. Load Artifacts and Product Details at Startup ---
model = None
user_map = None
product_map = None
interactions_df = None
user_item_sparse_matrix = None
product_details_df = None # To store product info (price, category)

try:
    # --- Load Model Artifacts ---
    model = joblib.load(os.path.join(ARTIFACTS_PATH, 'svd_model.joblib'))
    user_map = joblib.load(os.path.join(ARTIFACTS_PATH, 'svd_user_map.joblib'))
    product_map = joblib.load(os.path.join(ARTIFACTS_PATH, 'svd_product_map.joblib'))
    logging.info("✅ Recommendation model artifacts loaded successfully.")

    # --- Load and Prepare Product Details ---
    # This creates a lookup table for product information.
    products_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_products_cleaned_dataset.parquet'))
    order_items_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_order_items_cleaned_dataset.parquet'))
    
    # Calculate the average price for each product
    avg_price_df = order_items_df.groupby('product_id')['price'].mean().reset_index()
    
    # Merge with product category information
    product_details_df = pd.merge(
        products_df[['product_id', 'product_category_name']],
        avg_price_df,
        on='product_id',
        how='left'
    )
    product_details_df['price'].fillna(0, inplace=True) # Handle products that might not have a price yet
    logging.info("✅ Product details lookup table created.")


    # --- Recreate dataframes needed for the model ---
    orders_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_orders_cleaned_dataset.parquet'))
    df = pd.merge(orders_df, order_items_df, on='order_id')
    interactions_df = df.groupby(['customer_id', 'product_id']).size().reset_index(name='purchase_count')
    
    user_cat = pd.Categorical(interactions_df['customer_id'], categories=user_map.values())
    product_cat = pd.Categorical(interactions_df['product_id'], categories=product_map.values())
    
    row_indices = user_cat.codes
    col_indices = product_cat.codes
    
    user_item_sparse_matrix = csr_matrix((np.ones(len(interactions_df)), (row_indices, col_indices)), 
                                         shape=(len(user_map), len(product_map)))

    logging.info("✅ API is ready!")

except FileNotFoundError as e:
    logging.warning(f"⚠️ Error loading artifacts or data: {e}")
    model = None


# --- 3. Define the Prediction Endpoint ---
@router.get("/recommend/{customer_id}")
async def get_recommendations(customer_id: str):
    """
    Generates and returns top 10 product recommendations with details.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Recommendation model is not available.")

    user_map_inv = {v: k for k, v in user_map.items()}
    
    if customer_id not in user_map_inv:
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found.")

    try:
        user_index = user_map_inv[customer_id]
        user_vector = model.transform(user_item_sparse_matrix[user_index])[0]
        predicted_scores = np.dot(user_vector, model.components_)

        recommendations = pd.DataFrame({'product_id': list(product_map.values()), 'score': predicted_scores})

        bought_items = interactions_df[interactions_df['customer_id'] == customer_id]['product_id']
        final_recommendations = recommendations[~recommendations['product_id'].isin(bought_items)]
        
        top_10_recs = final_recommendations.sort_values(by='score', ascending=False).head(10)
        
        # --- Enrich Recommendations with Product Details ---
        # Merge the top 10 recommendations with our product details lookup table
        detailed_recs = pd.merge(
            top_10_recs,
            product_details_df,
            on='product_id',
            how='left'
        )
        # Fill any missing details for safety
        detailed_recs.fillna({'product_category_name': 'N/A', 'price': 0.0}, inplace=True)

        return {
            'customer_id': customer_id,
            # Convert the dataframe to a list of dictionaries for the JSON response
            'recommended_products': detailed_recs.to_dict('records')
        }
    except Exception as e:
        logging.error(f"Error during recommendation generation: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during recommendation.")

