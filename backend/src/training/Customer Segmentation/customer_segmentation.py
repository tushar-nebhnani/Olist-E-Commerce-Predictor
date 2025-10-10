import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pathlib import Path
import logging
import json
from datetime import timedelta
import joblib  # --- NEW --- Import joblib for saving the model

# --- 1. Configuration and Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
OUTPUTS_PATH = PROJECT_ROOT / "models" / "customer_segmentation"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)


def load_and_merge_data(path: Path) -> pd.DataFrame:
    """Loads and merges the necessary parquet files."""
    logging.info("Loading orders, customers, and payments data...")
    try:
        orders = pd.read_parquet(path / 'olist_orders_cleaned_dataset.parquet')
        customers = pd.read_parquet(path / 'olist_customers_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Ensure cleaned files are in '{path}'.")
        raise
        
    df = orders.merge(customers, on='customer_id')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary metrics."""
    logging.info("Calculating RFM metrics...")
    payments = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_payments_cleaned_dataset.parquet')
    order_payments = payments.groupby('order_id')['payment_value'].sum().reset_index()
    df = df.merge(order_payments, on='order_id')

    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    
    rfm_df = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'payment_value': 'Monetary'
    })
    return rfm_df

def engineer_features(df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers new features like Tenure and Inter-Purchase Time."""
    logging.info("Engineering advanced features...")
    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    
    customer_stats = df.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max', 'count'])
    
    customer_stats['Tenure'] = (snapshot_date - customer_stats['min']).dt.days
    
    purchase_time_diffs = df.sort_values(['customer_unique_id', 'order_purchase_timestamp']).groupby('customer_unique_id')['order_purchase_timestamp'].diff().dt.days
    avg_interpurchase_time = purchase_time_diffs.groupby(df['customer_unique_id']).mean()
    
    features_df = rfm_df.join(customer_stats['Tenure']).join(avg_interpurchase_time.rename('AvgInterpurchaseTime'))
    features_df['AvgInterpurchaseTime'].fillna(0, inplace=True)
    
    return features_df

def find_optimal_clusters(scaled_data: np.ndarray, max_k: int = 8) -> tuple[int, GaussianMixture]:
    """Finds the optimal number of clusters using GMM and Silhouette Score."""
    logging.info("Finding optimal number of clusters with GMM and Silhouette Score...")
    best_score = -1
    best_k = 2
    best_model = None

    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        labels = gmm.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        logging.info(f"For K={k}, Silhouette Score is {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = gmm
            
    logging.info(f"Optimal K found: {best_k} with Silhouette Score: {best_score:.4f}")
    return best_k, best_model

def analyze_and_present_clusters(features_df: pd.DataFrame) -> dict:
    """Analyzes clusters to assign personas and formats for frontend presentation."""
    logging.info("Automating cluster analysis for frontend...")
    
    numeric_cols = features_df.select_dtypes(include=np.number).drop(columns='Cluster').columns
    agg_funcs = {col: 'mean' for col in numeric_cols}
    agg_funcs['Cluster'] = 'count'
    
    summary = features_df.groupby('Cluster').agg(agg_funcs).rename(columns={'Cluster': 'Size'})

    summary['recency_rank'] = summary['Recency'].rank(ascending=True)
    summary['score'] = summary[[col for col in summary.columns if col != 'Recency']].sum(axis=1) - summary['recency_rank']
    
    personas = pd.cut(summary['score'].rank(method='first'), bins=len(summary), labels=False)
    persona_map = {
        len(summary)-1: "Champions",
        len(summary)-2: "Potential Loyalists",
        0: "At-Risk",
    }
    mid_tier_label = "Needs Attention"
    final_personas = {rank: persona_map.get(rank, mid_tier_label) for rank in range(len(summary))}
    
    summary['Persona'] = personas.map(final_personas)

    total_customers = summary['Size'].sum()
    analysis_output = {}
    for i, row in summary.iterrows():
        persona = row['Persona']
        analysis_output[str(i)] = {
            'persona': persona,
            'size': int(row['Size']),
            'percentage': f"{row['Size'] / total_customers:.1%}",
            'avg_recency': round(row['Recency'], 1),
            'avg_frequency': round(row['Frequency'], 2),
            'avg_monetary': round(row['Monetary'], 2),
            'avg_tenure_days': round(row.get('Tenure', 0), 1),
            'avg_interpurchase_time_days': round(row.get('AvgInterpurchaseTime', 0), 1),
        }
    return analysis_output

def perform_segmentation():
    """Main orchestrator for the advanced customer segmentation pipeline."""
    logging.info("--- Starting Advanced Customer Segmentation Pipeline ---")

    # 1. Load and Engineer
    base_df = load_and_merge_data(PROCESSED_DATA_PATH)
    rfm_df = calculate_rfm(base_df)
    features_df = engineer_features(base_df, rfm_df)
    
    # 2. Preprocess
    logging.info("Preprocessing data for clustering...")
    features_log = np.log1p(features_df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_log)
    
    # 3. Find Optimal Clusters and Predict
    optimal_k, best_gmm_model = find_optimal_clusters(features_scaled)
    features_df['Cluster'] = best_gmm_model.predict(features_scaled)

    # 4. Analyze and Save Results
    cluster_analysis = analyze_and_present_clusters(features_df)
    
    results_file = OUTPUTS_PATH / "customer_segments_advanced.csv"
    features_df.to_csv(results_file)
    logging.info(f"Advanced segment data saved to '{results_file}'")

    analysis_file = OUTPUTS_PATH / "segment_analysis_advanced.json"
    with open(analysis_file, 'w') as f:
        json.dump(cluster_analysis, f, indent=4)
    logging.info(f"Frontend-ready advanced analysis saved to '{analysis_file}'")
    
    # --- NEW --- Save the trained model object
    model_file = OUTPUTS_PATH / "customer_segmentation_gmm.joblib"
    joblib.dump(best_gmm_model, model_file)
    logging.info(f"Trained GMM model saved to '{model_file}'")
    
    logging.info("--- Advanced Customer Segmentation Finished ---")

if __name__ == '__main__':
    perform_segmentation()