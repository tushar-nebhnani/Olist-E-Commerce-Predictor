# Libraries to import
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pathlib import Path
import logging
import json
from datetime import timedelta

# --- 1. Configuration and Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
except IndexError:
    PROJECT_ROOT = Path('.').resolve()
    logging.warning(f"Could not determine project root. Using current directory: {PROJECT_ROOT}")

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
OUTPUTS_PATH = PROJECT_ROOT / "models" / "customer_segmentation"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)


# --- 2. Data Loading and Feature Engineering Functions ---

def load_and_merge_data(path: Path) -> pd.DataFrame:
    """Loads and merges the necessary pre-cleaned data files."""
    logging.info("Loading orders and customers data...")
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
    order_payments = payments.groupby('order_id')['capped_payment_value'].sum().reset_index().rename(columns={'capped_payment_value': 'Monetary'})
    df = df.merge(order_payments, on='order_id')
    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    rfm_df = df.groupby('customer_unique_id').agg({        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'Monetary': 'sum'
    }).rename(columns={'order_purchase_timestamp': 'Recency', 'order_id': 'Frequency'})
    return rfm_df

def engineer_features(df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Engineers Loyalty (Tenure) and Behavioral (Inter-Purchase Time) features."""
    logging.info("Engineering advanced features...")
    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    customer_stats = df.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max', 'count'])
    customer_stats['Tenure'] = (snapshot_date - customer_stats['min']).dt.days
    purchase_time_diffs = df.sort_values(['customer_unique_id', 'order_purchase_timestamp']).groupby('customer_unique_id')['order_purchase_timestamp'].diff().dt.days
    avg_interpurchase_time = purchase_time_diffs.groupby(df['customer_unique_id']).mean()
    features_df = rfm_df.join(customer_stats['Tenure']).join(avg_interpurchase_time.rename('AvgInterpurchaseTime'))
    features_df['AvgInterpurchaseTime'].fillna(0, inplace=True)
    return features_df


# --- 3. Modular Metric Calculation Functions ---

def calculate_wcss_for_kmeans(scaled_data: np.ndarray, max_k: int = 15) -> list:
    """Calculates Within-Cluster Sum of Squares (WCSS) for the Elbow Method."""
    logging.info("Calculating WCSS for K-Means Elbow Method...")
    wcss_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(scaled_data)
        wcss_scores.append(kmeans.inertia_)
        logging.info(f"K-Means | K={k}, WCSS={wcss_scores[-1]:.2f}")
    return wcss_scores

def calculate_silhouette_for_model(scaled_data: np.ndarray, model_class, max_k: int = 15) -> list:
    """Calculates Silhouette Scores for a given model class (KMeans or GMM)."""
    model_name = model_class.__name__
    logging.info(f"Calculating Silhouette Scores for {model_name}...")
    silhouette_scores = []
    for k in range(2, max_k + 1):
        model = model_class(n_components=k, random_state=42, n_init=10) if model_name == 'GaussianMixture' else model_class(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, labels)
        silhouette_scores.append(score)
        logging.info(f"{model_name} | K={k}, Silhouette Score={score:.4f}")
    return silhouette_scores

def calculate_bic_for_gmm(scaled_data: np.ndarray, max_k: int = 15) -> list:
    """Calculates Bayesian Information Criterion (BIC) specifically for GMM."""
    logging.info("Calculating BIC for GMM...")
    bic_scores = []
    for k in range(2, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        gmm.fit(scaled_data)
        bic_scores.append(gmm.bic(scaled_data))
        logging.info(f"GMM | K={k}, BIC={bic_scores[-1]:.2f}")
    return bic_scores

def calculate_k_distance_data(scaled_data: np.ndarray, min_samples_options: list) -> dict:
    """Calculates k-distance data needed for the DBSCAN eps elbow plot."""
    logging.info("Calculating K-Distance data for DBSCAN...")
    k_distance_results = {}
    for k in min_samples_options:
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(scaled_data)
        distances, _ = neighbors_fit.kneighbors(scaled_data)
        sorted_distances = np.sort(distances[:, k-1])
        k_distance_results[f"min_samples_{k}"] = list(sorted_distances)
        logging.info(f"DBSCAN | Calculated k-distances for min_samples={k}")
    return k_distance_results


# --- 4. Main Pre-computation Orchestrator ---

def precompute_validation_metrics():
    """Main orchestrator for the pre-computation pipeline."""
    logging.info("--- Starting Model Validation Pre-computation Pipeline ---")
    MAX_K = 15
    K_RANGE = list(range(2, MAX_K + 1))

    # Step 1: Data Prep
    base_df = load_and_merge_data(PROCESSED_DATA_PATH)
    rfm_df = calculate_rfm(base_df)
    features_df = engineer_features(base_df, rfm_df)
    
    # Step 2: Preprocessing
    logging.info("Preprocessing data: Log Transformation and Scaling...")
    features_log = np.log1p(features_df)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_log)
    
    # Step 3: Run all computations and store results
    logging.info("Starting calculation of all model-metric combinations...")
    validation_results = {"k_values": K_RANGE}

    # K-Means Calcs
    validation_results["kmeans"] = {
        "elbow": {"wcss_scores": calculate_wcss_for_kmeans(features_scaled, MAX_K)},
        "silhouette": {"scores": calculate_silhouette_for_model(features_scaled, KMeans, MAX_K)}
    }

    # GMM Calcs
    validation_results["gmm"] = {
        "bic": {"scores": calculate_bic_for_gmm(features_scaled, MAX_K)},
        "silhouette": {"scores": calculate_silhouette_for_model(features_scaled, GaussianMixture, MAX_K)}
    }

    # DBSCAN Calcs
    min_samples_to_test = [5, 10, 15]
    validation_results["dbscan"] = {
        "k_distance_plot_data": calculate_k_distance_data(features_scaled, min_samples_to_test),
        "min_samples_options": min_samples_to_test
    }

    # Step 4: Save the final JSON file
    results_file = OUTPUTS_PATH / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(validation_results, f, indent=4)
    logging.info(f"Dashboard-ready validation results saved to '{results_file}'")
    logging.info("--- Pre-computation Finished ---")


# --- 5. Execution Block ---
if __name__ == '__main__':
    precompute_validation_metrics()