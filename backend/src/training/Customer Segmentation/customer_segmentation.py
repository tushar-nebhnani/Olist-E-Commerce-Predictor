# Libraries to import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pathlib import Path
import logging
import json
from datetime import timedelta
import joblib

# --- 1. Configuration and Setup (MLOps Foundation) ---
# Establish a robust logging system for auditable production runs.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# For Production
# Define file paths using Python's 'pathlib' for OS-agnostic deployment.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
OUTPUTS_PATH = PROJECT_ROOT / "models" / "customer_segmentation"
# Ensure the output directory exists to store the final model artifact and results.
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

# Merging Of Dataset
def load_and_merge_data(path: Path) -> pd.DataFrame:
    """Loads and merges the necessary pre-cleaned data files."""
    logging.info("Loading orders, customers, and payments data...")
    # Using Parquet files is an MLOps choice for speed and correct data type preservation.
    try:
        orders = pd.read_parquet(path / 'olist_orders_cleaned_dataset.parquet')
        customers = pd.read_parquet(path / 'olist_customers_cleaned_dataset.parquet')
        # Note: Payments are merged later during RFM calculation to simplify the main dataframe merge.
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Ensure cleaned files are in '{path}'.")
        raise
        
    # Key Join: Links transactional behavior to the persistent customer identity.
    df = orders.merge(customers, on='customer_id')
    # Critical: Convert timestamp to datetime for accurate time-based feature calculation.
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary metrics."""
    logging.info("Calculating RFM metrics...")
    
    # Load payments data for Monetary aggregation
    payments = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_payments_cleaned_dataset.parquet')
    # Use the 'capped_payment_value' to aggregate, leveraging our outlier-controlled feature.
    order_payments = payments.groupby('order_id')['capped_payment_value'].sum().reset_index().rename(columns={'capped_payment_value': 'Monetary'})
    
    df = df.merge(order_payments, on='order_id')

    # Snapshot Date: The day *after* the last known transaction. This is the reference point for Recency.
    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    
    # RFM Calculation: Grouping by 'customer_unique_id' for accurate, de-duplicated customer metrics.
    rfm_df = df.groupby('customer_unique_id').agg({
        # Recency (R): Days since the last order. Low Recency is better.
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        # Frequency (F): Number of unique orders. High Frequency is better.
        'order_id': 'nunique',
        # Monetary (M): Total value of all purchases (using the capped, stable feature).
        'Monetary': 'sum' 
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency'
    })
    return rfm_df

def engineer_features(df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers Loyalty (Tenure) and Behavioral (Inter-Purchase Time) features."""
    logging.info("Engineering advanced features...")
    
    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    
    # Calculate initial stats for Tenure
    customer_stats = df.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max', 'count'])
    
    # Tenure (Customer Age): Days between the first purchase and the snapshot date. 
    # CRITICAL for distinguishing new vs. loyal customers with similar RFM scores.
    customer_stats['Tenure'] = (snapshot_date - customer_stats['min']).dt.days
    
    # AvgInterpurchaseTime: Average time (in days) between consecutive orders.
    # Key behavioral predictor of the next purchase or churn risk.
    purchase_time_diffs = df.sort_values(['customer_unique_id', 'order_purchase_timestamp']).groupby('customer_unique_id')['order_purchase_timestamp'].diff().dt.days
    avg_interpurchase_time = purchase_time_diffs.groupby(df['customer_unique_id']).mean()
    
    # Final Merge
    features_df = rfm_df.join(customer_stats['Tenure']).join(avg_interpurchase_time.rename('AvgInterpurchaseTime'))
    
    # Imputation: Fill NaN for single-purchase customers with 0. 
    # ML RATIONALE: This forces all one-time buyers into a sharp, distinct cluster point,
    # improving the separability of the "At-Risk/Transactional" segment.
    features_df['AvgInterpurchaseTime'].fillna(0, inplace=True)
    
    return features_df

# Bic Calculation to ensure the correct number of segements
def plot_bic_elbow(bic_values: list, max_k: int):
    """Generates and saves the BIC Elbow Plot for K optimization."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_k + 1), bic_values, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Bayesian Information Criterion (BIC)")
    plt.title("GMM Cluster Selection: BIC Elbow Plot")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_file = OUTPUTS_PATH / "bic_elbow_plot.png"
    plt.savefig(plot_file)
    logging.info(f"BIC Elbow Plot saved to '{plot_file}'")
    plt.close()

# TO check the number of optimal clusters
def find_optimal_clusters(scaled_data: np.ndarray, max_k: int = 8) -> tuple[int, GaussianMixture]:
    """Finds the optimal K using GMM and the Bayesian Information Criterion (BIC)."""
    logging.info("Finding optimal K with GMM and BIC...")
    # GMM RATIONALE: Chosen over K-Means for its ability to model non-spherical and overlapping
    # clusters (the reality of customer behavior).
    
    bic_values = []
    best_bic = np.inf # BIC minimization
    best_k = 2
    best_model = None
    
    # Iterate K from 2 up to max_k.
    for k in range(2, max_k + 1):
        # GMM CONFIGURATION: n_init=10 for robust initialization, random_state for reproducibility.
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(scaled_data)
        
        # BIC RATIONALE: BIC is the theoretically superior metric for GMM, as it penalizes
        # model complexity (higher K) and ensures the chosen K is statistically justified.
        bic = gmm.bic(scaled_data)
        bic_values.append(bic)
        
        logging.info(f"For K={k}, BIC is {bic:.2f}")
        
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = gmm
            
    plot_bic_elbow(bic_values, max_k)
    logging.info(f"Optimal K found: {best_k} (Minimum BIC: {best_bic:.2f})")
    
    # Optional: Validate K using Silhouette Score (for interpretability/defense)
    labels = best_model.predict(scaled_data)
    sil_score = silhouette_score(scaled_data, labels)
    logging.info(f"Optimal Model Silhouette Score: {sil_score:.4f}")
    
    return best_k, best_model

def analyze_and_present_clusters(features_df: pd.DataFrame) -> dict:
    """Analyzes clusters to assign custom personas and formats for business consumption."""
    logging.info("Automating cluster analysis and persona assignment...")
    
    # Step 1: Calculate the mean of all features by cluster to profile each segment.
    numeric_cols = features_df.select_dtypes(include=np.number).drop(columns='Cluster').columns
    agg_funcs = {col: 'mean' for col in numeric_cols}
    agg_funcs['Cluster'] = 'count'
    summary = features_df.groupby('Cluster').agg(agg_funcs).rename(columns={'Cluster': 'Size'})

    # Step 2: Custom Persona Scoring for Business Actionability
    # This composite score rewards cumulative value and punishes current inactivity.
    summary['recency_rank'] = summary['Recency'].rank(ascending=True) # Lowest Recency (most active) gets highest rank
    
    # Score calculation: Sum of all positive metrics MINUS a penalty for Recency (inactivity).
    summary['score'] = summary['Frequency'] + summary['Monetary'] + summary['Tenure'] - (summary['Recency'] * 0.1)
    
    # Step 3: Assign clear, static Personas based on the score ranking.
    score_ranks = summary['score'].rank(method='first', ascending=True)
    
    num_clusters = len(summary)
    persona_labels = [
        "Champions" if rank == num_clusters else
        "At-Risk" if rank == 1 else
        "Potential Loyalists" if rank == num_clusters - 1 else
        "Needs Attention"
        for rank in score_ranks
    ]
    summary['Persona'] = persona_labels
    
    # Step 4: Final Formatting for Presentation/API
    total_customers = summary['Size'].sum()
    analysis_output = {}
    for i, row in summary.iterrows():
        analysis_output[str(i)] = {
            'persona': row['Persona'],
            'size': int(row['Size']),
            'percentage': f"{row['Size'] / total_customers:.1%}",
            'avg_recency_days': round(row['Recency'], 1),
            'avg_frequency': round(row['Frequency'], 2),
            'avg_monetary': round(row['Monetary'], 2),
            'avg_tenure_days': round(row.get('Tenure', 0), 1),
            'avg_interpurchase_time_days': round(row.get('AvgInterpurchaseTime', 0), 1),
        }
    return analysis_output

def perform_segmentation():
    """Main orchestrator for the advanced customer segmentation pipeline."""
    logging.info("--- Starting Advanced Customer Segmentation Pipeline ---")

    # 1. Load and Engineer Features
    base_df = load_and_merge_data(PROCESSED_DATA_PATH)
    rfm_df = calculate_rfm(base_df)
    features_df = engineer_features(base_df, rfm_df)
    
    # 2. Preprocess Data (Crucial for distance-based clustering models)
    logging.info("Preprocessing data: Log Transformation and Scaling...")
    # Log Transformation (np.log1p): Mitigates the extreme skew of value-based features 
    # (Monetary, Tenure, Frequency) to make them more Gaussian-like for GMM.
    features_log = np.log1p(features_df)
    
    # StandardScaler: Normalizes features to mean 0, std dev 1. 
    # RATIONALE: Prevents high-magnitude features (Monetary) from dominating the clustering space.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_log)
    
    # 3. Find Optimal Clusters and Predict
    # The scaler is fitted on the log-transformed data.
    optimal_k, best_gmm_model = find_optimal_clusters(features_scaled)
    features_df['Cluster'] = best_gmm_model.predict(features_scaled)

    # 4. Analyze and Save Results
    cluster_analysis = analyze_and_present_clusters(features_df)
    
    # Production Artifact 1: CSV of all customers with their segment label (for CRM database integration).
    results_file = OUTPUTS_PATH / "customer_segments_advanced.csv"
    features_df.to_csv(results_file)
    logging.info(f"Advanced segment data saved to '{results_file}'")

    # Production Artifact 2: JSON file for front-end dashboard or BI tool consumption.
    analysis_file = OUTPUTS_PATH / "segment_analysis_advanced.json"
    with open(analysis_file, 'w') as f:
        json.dump(cluster_analysis, f, indent=4)
    logging.info(f"Frontend-ready advanced analysis saved to '{analysis_file}'")
    
    # Production Artifact 3: Save the trained GMM model and the fitted scaler object.
    # CRITICAL MLOPS STEP: Allows the model to be loaded for instant inference on new data.
    model_file = OUTPUTS_PATH / "customer_segmentation_gmm.joblib"
    joblib.dump(best_gmm_model, model_file)
    logging.info(f"Trained GMM model saved to '{model_file}'")
    
    scaler_file = OUTPUTS_PATH / "scaler_gmm.joblib"
    joblib.dump(scaler, scaler_file)
    logging.info(f"Fitted StandardScaler saved to '{scaler_file}'")
    
    logging.info("--- Advanced Customer Segmentation Finished ---")

if __name__ == '__main__':
    # Ensure the script is only run when executed directly.
    perform_segmentation()