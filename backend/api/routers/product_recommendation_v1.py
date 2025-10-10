# File: backend/api/routers/product_recommendation_v1.py

import os
import joblib
import pandas as pd
from fastapi import APIRouter, Request, HTTPException, FastAPI
import logging
import numpy as np
from scipy.sparse import csr_matrix

# --- Router Setup ---
router = APIRouter()

# --- Artifact Loading Function ---
async def load_recommendation_models(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.info("--- 🚀 STARTUP: Began loading recommendation models. ---")
    
    try:
        # Define paths
        ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
        API_DIR = os.path.dirname(ROUTER_DIR)
        BACKEND_DIR = os.path.dirname(API_DIR)
        ARTIFACTS_PATH = os.path.join(BACKEND_DIR, "models", "Product Recommendation")
        DATA_PATH = os.path.join(BACKEND_DIR, "data", "processed")

        # Load models and maps
        app.state.svd_model = joblib.load(os.path.join(ARTIFACTS_PATH, 'svd_model.joblib'))
        app.state.user_map = joblib.load(os.path.join(ARTIFACTS_PATH, 'svd_user_map.joblib'))
        app.state.product_map = joblib.load(os.path.join(ARTIFACTS_PATH, 'svd_product_map.joblib'))
        app.state.user_id_to_index = {v: k for k, v in app.state.user_map.items()}

        # Load and prepare a richer products_df
        logging.info("  [LOAD] Loading base data files...")
        orders_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_orders_cleaned_dataset.parquet'))
        order_items_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_order_items_cleaned_dataset.parquet'))
        products_info_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_products_cleaned_dataset.parquet'))
        translation_path = os.path.join(DATA_PATH, 'category_name_translation_cleaned_dataset.parquet')
        category_translations_df = pd.read_parquet(translation_path)
        
        logging.info("  [PROC] Merging dataframes to create detailed product list...")

        # 1. Get the average price for each product
        avg_price_df = order_items_df.groupby('product_id')['price'].mean().reset_index()

        # 2. Merge product info with translations
        products_with_english_names = pd.merge(
            products_info_df,
            category_translations_df,
            on='product_category_name',
            how='left'
        )

        # 3. Merge the result with the average price
        final_products_df = pd.merge(
            products_with_english_names,
            avg_price_df,
            on='product_id',
            how='left'
        )

        # 4. Robust fallbacks for all key fields
        final_products_df['product_category_name_english'] = final_products_df['product_category_name_english'].fillna(
            final_products_df['product_category_name']
        ).fillna('general')
        final_products_df['price'] = final_products_df['price'].fillna(0)

        # Store the final, enriched dataframe
        app.state.products_df = final_products_df
        logging.info("  [OK]   ...successfully created detailed product dataframe.")
        
        # Load interactions and build the sparse matrix
        merged_df = pd.merge(orders_df, order_items_df, on='order_id')
        app.state.interactions_df = merged_df[['customer_id', 'product_id']].drop_duplicates()
        
        logging.info("  [PROC] Building user-item sparse matrix...")
        user_ids_in_order = [app.state.user_map[i] for i in range(len(app.state.user_map))]
        product_ids_in_order = [app.state.product_map[i] for i in range(len(app.state.product_map))]
        user_cat = pd.Categorical(app.state.interactions_df['customer_id'], categories=user_ids_in_order, ordered=True)
        product_cat = pd.Categorical(app.state.interactions_df['product_id'], categories=product_ids_in_order, ordered=True)
        
        # Handle potential NaNs in categorical codes
        user_codes = user_cat.codes
        product_codes = product_cat.codes
        valid_indices = (user_codes != -1) & (product_codes != -1)

        app.state.user_item_sparse_matrix = csr_matrix(
            (np.ones(np.sum(valid_indices)), (user_codes[valid_indices], product_codes[valid_indices])),
            shape=(len(app.state.user_map), len(app.state.product_map))
        )
        logging.info("  [OK]   ...sparse matrix built successfully.")

    except Exception as e:
        logging.error(f"  [CRASH] ❌ A fatal error occurred during model loading: {e}", exc_info=True)
        app.state.recommendation_ready = False
    else:
        app.state.recommendation_ready = True
        logging.info("--- ✅✅✅ STARTUP: All recommendation artifacts loaded successfully! ---")

# --- API Endpoints ---
def check_model_loaded(request: Request):
    if not getattr(request.app.state, 'recommendation_ready', False):
        raise HTTPException(status_code=503, detail="Service unavailable: Recommendation model failed to load.")

@router.get("/{customer_id}", tags=["Product Recommendation"])
async def get_recommendations(customer_id: str, request: Request):
    check_model_loaded(request)
    user_id_to_index = request.app.state.user_id_to_index
    if customer_id not in user_id_to_index:
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' not found.")

    try:
        user_index = user_id_to_index[customer_id]
        user_vector = request.app.state.svd_model.transform(request.app.state.user_item_sparse_matrix[user_index])
        item_matrix = request.app.state.svd_model.components_
        predicted_scores = np.dot(user_vector, item_matrix).flatten()
        
        recs_df = pd.DataFrame({'product_id': list(request.app.state.product_map.values()), 'score': predicted_scores})
        bought_items = request.app.state.interactions_df[request.app.state.interactions_df['customer_id'] == customer_id]['product_id']
        final_recs_df = recs_df[~recs_df['product_id'].isin(bought_items)]
        top_10_recs = final_recs_df.sort_values(by='score', ascending=False).head(10)
        
        detailed_recs = pd.merge(top_10_recs, request.app.state.products_df, on='product_id', how='left')
        
        # Handle potential nulls in all numeric columns before sending the response
        numeric_cols = [
            'price', 'product_name_length', 'product_description_length', 
            'product_photos_qty', 'product_weight_g', 'product_length_cm', 
            'product_height_cm', 'product_width_cm'
        ]
        for col in numeric_cols:
            if col in detailed_recs.columns:
                detailed_recs[col] = detailed_recs[col].fillna(0)

        return {'customer_id': customer_id, 'recommended_products': detailed_recs.to_dict('records')}
        
    except Exception as e:
        logging.error(f"Error during recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during recommendation generation.")