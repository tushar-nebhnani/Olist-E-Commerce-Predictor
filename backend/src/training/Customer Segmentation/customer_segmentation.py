import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pathlib import Path
import logging
import json
from datetime import timedelta
import joblib  # This is crucial for PRODUCTION: allows saving/loading the trained model.

# --- 1. Configuration and Setup ---
# We establish a robust logging system and clear file path structure 
# to ensure the entire process is auditable and easily deployable in production.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
OUTPUTS_PATH = PROJECT_ROOT / "models" / "customer_segmentation"
# Actionable: Create output directory to store results and the model object.
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)


def load_and_merge_data(path: Path) -> pd.DataFrame:
    """Loads and merges the necessary parquet files."""
    logging.info("Loading orders, customers, and payments data...")
    # We use pre-cleaned Parquet files for speed and data type efficiency.
    try:
        orders = pd.read_parquet(path / 'olist_orders_cleaned_dataset.parquet')
        customers = pd.read_parquet(path / 'olist_customers_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Ensure cleaned files are in '{path}'.")
        raise
        
    # Key Join: Merging orders and customer data via 'customer_id' 
    # This links transaction behavior to the customer entity.
    df = orders.merge(customers, on='customer_id')
    # Critical: Convert timestamp to datetime object for accurate time-based calculations (Recency, Tenure).
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary metrics."""
    logging.info("Calculating RFM metrics...")
    
    # Need to aggregate payment values first as one order can have multiple payment records.
    payments = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_payments_cleaned_dataset.parquet')
    order_payments = payments.groupby('order_id')['payment_value'].sum().reset_index()
    # Merge Monetary Value back to the main DataFrame
    df = df.merge(order_payments, on='order_id')

    # Snapshot Date: The day *after* the last known transaction. 
    # This is the reference point for calculating Recency (Days since last purchase).
    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    
    # RFM Calculation: Grouping by 'customer_unique_id' for accurate, de-duplicated customer metrics.
    rfm_df = df.groupby('customer_unique_id').agg({
        # Recency (R): Days since the last order. Low Recency is better.
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        # Frequency (F): Number of unique orders. High Frequency is better.
        'order_id': 'nunique',
        # Monetary (M): Total value of all purchases. High Monetary is better.
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
    # RFM alone is good, but adding Tenure (Loyalty) and AvgInterpurchaseTime (Behavioral Predictor)
    # allows for a more stable and predictive segmentation model.
    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    
    # Calculate initial stats for Tenure
    customer_stats = df.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max', 'count'])
    
    # Tenure (Customer Age): Days between the first purchase ('min') and the snapshot date. 
    # Crucial for distinguishing New vs. Loyal customers, even if their RFM scores are temporarily similar.
    customer_stats['Tenure'] = (snapshot_date - customer_stats['min']).dt.days
    
    # AvgInterpurchaseTime: The average time (in days) between consecutive orders.
    # This is a key behavioral metric for predicting the next purchase or identifying abnormal gaps.
    purchase_time_diffs = df.sort_values(['customer_unique_id', 'order_purchase_timestamp']).groupby('customer_unique_id')['order_purchase_timestamp'].diff().dt.days
    avg_interpurchase_time = purchase_time_diffs.groupby(df['customer_unique_id']).mean()
    
    # Final Merge
    features_df = rfm_df.join(customer_stats['Tenure']).join(avg_interpurchase_time.rename('AvgInterpurchaseTime'))
    
    # Imputation: Fill NaN for single-purchase customers with 0. 
    # This creates a strong, distinct point in the feature space for single-time buyers.
    features_df['AvgInterpurchaseTime'].fillna(0, inplace=True)
    
    return features_df

def find_optimal_clusters(scaled_data: np.ndarray, max_k: int = 8) -> tuple[int, GaussianMixture]:
    """Finds the optimal number of clusters using GMM and Silhouette Score."""
    logging.info("Finding optimal number of clusters with GMM and Silhouette Score...")
    # We chose Gaussian Mixture Model (GMM) over K-Means because GMM is probabilistic, 
    # better handles non-spherical clusters, and provides a 'soft' cluster assignment, 
    # reflecting the reality of overlapping customer behaviors.
    best_score = -1
    best_k = 2
    best_model = None

    # Iteratively test K from 2 up to the specified max_k (default 8).
    for k in range(2, max_k + 1):
        # GMM Configuration: random_state=42 for reproducibility; n_init=10 to find a robust fit.
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        labels = gmm.fit_predict(scaled_data)
        # Silhouette Score: A common metric measuring how similar an object is to its own cluster 
        # compared to other clusters. Higher score indicates better separation.
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
    
    # Step 1: Calculate the mean of all features by cluster to understand the *profile* of each segment.
    numeric_cols = features_df.select_dtypes(include=np.number).drop(columns='Cluster').columns
    agg_funcs = {col: 'mean' for col in numeric_cols}
    agg_funcs['Cluster'] = 'count' # Use the cluster column to count the size of the segment
    
    summary = features_df.groupby('Cluster').agg(agg_funcs).rename(columns={'Cluster': 'Size'})

    # Step 2: Custom Persona Scoring for Business Actionability
    # The scoring mechanism ranks clusters based on overall value, prioritizing high F, M, and Tenure.
    summary['recency_rank'] = summary['Recency'].rank(ascending=True) # Lowest Recency (most recent) gets highest rank
    
    # Custom Score: Sum of all valuable metrics (F, M, Tenure, etc.) MINUS the Recency Rank penalty.
    # This ensures that a customer group is only a "Champion" if they are high-value AND active (low Recency).
    summary['score'] = summary[[col for col in summary.columns if col != 'Recency']].sum(axis=1) - summary['recency_rank']
    
    # Step 3: Assign clear, actionable Personas based on the score ranking.
    # The highest score gets the most positive persona (Champions), and the lowest gets the most concerning (At-Risk).
    personas = pd.cut(summary['score'].rank(method='first'), bins=len(summary), labels=False)
    persona_map = {
        len(summary)-1: "Champions",         # Highest Score/Rank (Most Valuable/Active)
        len(summary)-2: "Potential Loyalists", # Second Highest Score/Rank
        0: "At-Risk",                       # Lowest Score/Rank (Highest Churn Risk)
    }
    mid_tier_label = "Needs Attention" # Generic label for all intermediate segments
    final_personas = {rank: persona_map.get(rank, mid_tier_label) for rank in range(len(summary))}
    
    summary['Persona'] = personas.map(final_personas)

    # Step 4: Final Formatting for Presentation/API
    total_customers = summary['Size'].sum()
    analysis_output = {}
    for i, row in summary.iterrows():
        persona = row['Persona']
        analysis_output[str(i)] = {
            # Business Segment Name
            'persona': persona,
            # Segment Size (for resource allocation)
            'size': int(row['Size']),
            'percentage': f"{row['Size'] / total_customers:.1%}",
            # Key Feature Metrics (Averaged by segment for profiling)
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
    
    # 2. Preprocess (The necessary step for distance-based clustering)
    logging.info("Preprocessing data for clustering...")
    # Log Transformation (np.log1p): Mitigates the extreme skew of RFM and other value-based features, 
    # making the distributions more Gaussian-like for the GMM.
    features_log = np.log1p(features_df)
    # StandardScaler: Normalizes all features to a mean of 0 and standard deviation of 1.
    # This prevents high-magnitude features (e.g., Monetary) from dominating the clustering process.
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_log)
    
    # 3. Find Optimal Clusters and Predict
    optimal_k, best_gmm_model = find_optimal_clusters(features_scaled)
    # Final step: Assign the predictive cluster label back to the original, understandable DataFrame.
    features_df['Cluster'] = best_gmm_model.predict(features_scaled)

    # 4. Analyze and Save Results
    cluster_analysis = analyze_and_present_clusters(features_df)
    
    # CSV of all customers with their features and final cluster label (for database integration/CRM).
    results_file = OUTPUTS_PATH / "customer_segments_advanced.csv"
    features_df.to_csv(results_file)
    logging.info(f"Advanced segment data saved to '{results_file}'")

    # JSON file containing the segment summaries (ideal for a BI tool or front-end dashboard).
    analysis_file = OUTPUTS_PATH / "segment_analysis_advanced.json"
    with open(analysis_file, 'w') as f:
        json.dump(cluster_analysis, f, indent=4)
    logging.info(f"Frontend-ready advanced analysis saved to '{analysis_file}'")
    
    # Production Readiness: Save the trained GMM model itself using Joblib.
    # This allows the model to be loaded and used to score new incoming customers without re-training.
    model_file = OUTPUTS_PATH / "customer_segmentation_gmm.joblib"
    joblib.dump(best_gmm_model, model_file)
    logging.info(f"Trained GMM model saved to '{model_file}'")
    
    logging.info("--- Advanced Customer Segmentation Finished ---")

if __name__ == '__main__':
    perform_segmentation()