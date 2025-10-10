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

        # Load base dataframes
        orders_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_orders_cleaned_dataset.parquet'))
        order_items_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_order_items_cleaned_dataset.parquet'))
        products_info_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_products_cleaned_dataset.parquet'))
        
        # --- FIX: Use your exact filename and the correct read function ---
        logging.info("  [LOAD] Loading category name translation parquet...")
        translation_path = os.path.join(DATA_PATH, 'category_name_translation_cleaned_dataset.parquet') # <-- UPDATED FILENAME
        category_translations_df = pd.read_parquet(translation_path) # <-- UPDATED FUNCTION
        logging.info("  [OK]   ...loaded translation file.")

        # --- MERGE DATAFRAMES TO ADD ENGLISH NAMES ---
        logging.info("  [PROC] Merging dataframes to add English names...")
        
        products_with_english_names = pd.merge(
            products_info_df[['product_id', 'product_category_name']],
            category_translations_df,
            on='product_category_name',
            how='left'
        )
        final_products_df = pd.merge(
            order_items_df[['product_id', 'price']],
            products_with_english_names,
            on='product_id'
        ).drop_duplicates(subset='product_id')
        final_products_df['product_category_name_english'] = final_products_df['product_category_name_english'].fillna(
            final_products_df['product_category_name']
        )
        final_products_df['product_category_name_english'] = final_products_df['product_category_name_english'].fillna('general')
        app.state.products_df = final_products_df
        logging.info("  [OK]   ...successfully merged and processed product dataframes.")

        # --- (The rest of the function remains the same) ---
        merged_df = pd.merge(orders_df, order_items_df, on='order_id')
        app.state.interactions_df = merged_df[['customer_id', 'product_id']].drop_duplicates()
        logging.info("  [PROC] Building user-item sparse matrix...")
        user_ids_in_order = [app.state.user_map[i] for i in range(len(app.state.user_map))]
        product_ids_in_order = [app.state.product_map[i] for i in range(len(app.state.product_map))]
        user_cat = pd.Categorical(app.state.interactions_df['customer_id'], categories=user_ids_in_order)
        product_cat = pd.Categorical(app.state.interactions_df['product_id'], categories=product_ids_in_order)
        app.state.user_item_sparse_matrix = csr_matrix(
            (np.ones(len(app.state.interactions_df)), (user_cat.codes, product_cat.codes)),
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
        
        # --- FIX: USE THE ENRICHED products_df FOR THE FINAL MERGE ---
        detailed_recs = pd.merge(top_10_recs, request.app.state.products_df, on='product_id', how='left')
        
        # Ensure data types are correct for the JSON response
        detailed_recs['price'] = detailed_recs['price'].fillna(0.0).astype(float)
        # We now use the english name, but it's good practice to handle the original too
        detailed_recs['product_category_name'] = detailed_recs['product_category_name'].fillna('N/A').astype(str)
        detailed_recs['product_category_name_english'] = detailed_recs['product_category_name_english'].fillna('N/A').astype(str)

        return {'customer_id': customer_id, 'recommended_products': detailed_recs.to_dict('records')}
    except Exception as e:
        logging.error(f"Error during recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during recommendation generation.")