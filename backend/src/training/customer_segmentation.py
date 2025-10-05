import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import timedelta

def perform_customer_segmentation():
    """
    Performs RFM analysis and K-Means clustering to segment customers.
    """
    print("--- Starting Customer Segmentation ---")

    # --- 1. Define Paths and Load Data ---
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    
    print("Step 1: Loading orders, customers, and payments data...")
    try:
        orders = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_orders_cleaned_dataset.parquet')
        customers = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_customers_cleaned_dataset.parquet')
        payments = pd.read_parquet(PROCESSED_DATA_PATH / 'olist_order_payments_cleaned_dataset.parquet')
    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure cleaned files are in '{PROCESSED_DATA_PATH}'.")
        return

    # Merge the datasets
    df = orders.merge(customers, on='customer_id').merge(payments, on='order_id')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    # --- 2. RFM (Recency, Frequency, Monetary) Analysis ---
    print("Step 2: Performing RFM Analysis...")
    
    # Set a "snapshot date" to calculate recency from. This is the day after the last order.
    snapshot_date = df['order_purchase_timestamp'].max() + timedelta(days=1)
    
    # Group by each unique customer
    rfm_df = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    })

    # Rename columns for clarity
    rfm_df.rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'payment_value': 'Monetary'
    }, inplace=True)

    print("RFM calculation complete. Sample of RFM data:")
    print(rfm_df.head())

    # --- 3. Preprocess Data for Clustering ---
    print("\nStep 3: Preprocessing data for K-Means...")
    
    # Handle skewness with log transformation
    rfm_log = np.log1p(rfm_df)
    
    # Scale the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # --- 4. Find the Optimal Number of Clusters (K) using the Elbow Method ---
    print("\nStep 4: Finding optimal K using the Elbow Method...")
    
    inertia = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(rfm_scaled)
        inertia[k] = kmeans.inertia_

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 5))
    plt.plot(list(inertia.keys()), list(inertia.values()), 'o-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.xticks(range(1, 11))
    plt.grid(True)
    print("Displaying Elbow Method plot. Close the plot to continue...")
    plt.show()

    # --- 5. Apply K-Means Clustering ---
    # Based on the elbow plot, choose the best K (often 4 for this dataset)
    optimal_k = 4
    print(f"\nStep 5: Applying K-Means with K={optimal_k}...")
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # --- 6. Analyze the Clusters ---
    print("\nStep 6: Analyzing cluster characteristics...")
    
    # Calculate the average RFM values for each cluster
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(2)
    
    print("Cluster Summary:")
    print(cluster_summary)
    
    print("\n--- Customer Segmentation Finished ---")

if __name__ == '__main__':
    perform_customer_segmentation()