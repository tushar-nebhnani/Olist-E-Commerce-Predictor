import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def load_data(path_dict):
    """Loads all necessary parquet files into pandas DataFrames."""
    data = {}
    for key, path in path_dict.items():
        try:
            # Changed to read parquet files as requested
            data[key] = pd.read_parquet(path)
        except FileNotFoundError:
            print(f"Error: The file at {path} was not found.")
            return None
    return data

def preprocess_data(data):
    """Merges and preprocesses the Olist dataset."""
    # Merge datasets to get a comprehensive view
    orders_customers = pd.merge(data['orders'], data['customers'], on='customer_id')
    orders_payments = pd.merge(orders_customers, data['payments'], on='order_id')
    order_items_products = pd.merge(data['order_items'], data['products'], on='product_id')
    full_data = pd.merge(orders_payments, order_items_products, on='order_id')

    # Updated date columns to match your new schema
    date_columns = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_deadline']
    for col in date_columns:
        full_data[col] = pd.to_datetime(full_data[col], errors='coerce')

    full_data.dropna(subset=['order_purchase_timestamp', 'customer_unique_id'], inplace=True)

    return full_data

def calculate_rfm(df, snapshot_date):
    """Calculates Recency, Frequency, and Monetary values for each customer based on a snapshot date."""
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).rename(columns={'order_purchase_timestamp': 'Recency', 'order_id': 'Frequency', 'price': 'MonetaryValue'})

    return rfm

def create_churn_feature(rfm):
    """Creates the target variable 'Churn' based on recency."""
    # We'll use the 75th percentile of recency as the threshold.
    churn_threshold = rfm['Recency'].quantile(0.75)
    rfm['Churn'] = (rfm['Recency'] > churn_threshold).astype(int)
    return rfm

def train_churn_model(rfm_df):
    """Trains a RandomForestClassifier to predict customer churn."""
    # FIX: Removed 'Recency' from features to prevent data leakage.
    X = rfm_df[['Frequency', 'MonetaryValue']]
    y = rfm_df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("--- Churn Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Predict probability using the same features
    rfm_df['Churn_Probability'] = model.predict_proba(X)[:, 1]
    return model, rfm_df

def train_clv_model(df):
    """Prepares data and trains a regression model to predict Customer Lifetime Value."""
    print("\n--- Training CLV Prediction Model ---")
    
    # Define calibration and observation periods.
    df = df.sort_values('order_purchase_timestamp')
    split_point = df['order_purchase_timestamp'].max() - pd.DateOffset(days=90)
    
    calibration_df = df[df['order_purchase_timestamp'] < split_point]
    observation_df = df[df['order_purchase_timestamp'] >= split_point]

    calibration_rfm = calculate_rfm(calibration_df, split_point)

    observation_value = observation_df.groupby('customer_unique_id')['price'].sum().rename('Future_CLV')
    
    clv_df = pd.merge(calibration_rfm, observation_value, on='customer_unique_id', how='left')
    clv_df['Future_CLV'].fillna(0, inplace=True)

    # FIX: Apply log transformation to skewed features to improve model performance.
    clv_df_transformed = clv_df.copy()
    for col in ['Recency', 'Frequency', 'MonetaryValue']:
        clv_df_transformed[col] = np.log1p(clv_df_transformed[col])

    X = clv_df_transformed[['Recency', 'Frequency', 'MonetaryValue']]
    y = clv_df['Future_CLV']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"CLV Model Mean Absolute Error (MAE): {mae:.2f}")
    print(f"CLV Model R-squared (R²): {r2:.2f}")

    return model

def customer_segmentation(rfm_df):
    """Segments customers using K-Means clustering on RFM values."""
    rfm_scaled = StandardScaler().fit_transform(rfm_df[['Recency', 'Frequency', 'MonetaryValue']])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm_df['Segment'] = kmeans.fit_predict(rfm_scaled)

    # FIX: More robust segment naming logic
    segment_stats = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'MonetaryValue']].mean()
    
    # Identify segments based on characteristics
    high_value_seg = segment_stats.sort_values(['MonetaryValue', 'Frequency'], ascending=[False, False]).index[0]
    at_risk_seg = segment_stats.sort_values('Recency', ascending=False).index[0]
    new_cust_seg = segment_stats.sort_values(['Recency', 'Frequency'], ascending=[True, True]).index[0]
    
    # Create map
    segment_map = {high_value_seg: 'High Value - Loyal'}
    
    # Assign At Risk, making sure it's not the same as high value
    if at_risk_seg not in segment_map:
        segment_map[at_risk_seg] = 'At Risk / Churned'

    # Assign New Customers, making sure it's not already assigned
    if new_cust_seg not in segment_map:
       segment_map[new_cust_seg] = 'New Customers'

    # Assign the remaining segment
    for seg_id in segment_stats.index:
        if seg_id not in segment_map:
            segment_map[seg_id] = 'Potential to be Loyal'
            
    rfm_df['Segment_Name'] = rfm_df['Segment'].map(segment_map)
    return rfm_df, segment_stats


def main():
    """Main function to run the customer analytics pipeline."""
    # Updated data paths to use your parquet files
    data_paths = {
        'customers': 'D:/Data Science/CaseStudy ML/Olist-E-Commerce-Predictor-/backend/data/processed/olist_customers_cleaned_dataset.parquet',
        'orders': 'D:/Data Science/CaseStudy ML/Olist-E-Commerce-Predictor-/backend/data/processed/olist_orders_cleaned_dataset.parquet',
        'payments': 'D:/Data Science/CaseStudy ML/Olist-E-Commerce-Predictor-/backend/data/processed/olist_order_payments_cleaned_dataset.parquet',
        'order_items': 'D:/Data Science/CaseStudy ML/Olist-E-Commerce-Predictor-/backend/data/processed/olist_order_items_cleaned_dataset.parquet',
        'products': 'D:/Data Science/CaseStudy ML/Olist-E-Commerce-Predictor-/backend/data/processed/olist_products_cleaned_dataset.parquet',
    }

    print("Loading data...")
    olist_data = load_data(data_paths)
    if olist_data is None:
        print("\nCould not load data. Please check the file paths.")
        return

    print("Preprocessing data...")
    processed_data = preprocess_data(olist_data)

    snapshot_date_full = processed_data['order_purchase_timestamp'].max() + pd.DateOffset(days=1)
    rfm_full_data = calculate_rfm(processed_data, snapshot_date_full)

    print("Creating churn feature...")
    rfm_with_churn = create_churn_feature(rfm_full_data)

    print("Training churn prediction model...")
    _, rfm_with_churn = train_churn_model(rfm_with_churn)
    
    clv_model = train_clv_model(processed_data)

    # FIX: Apply the same log transformation before prediction
    rfm_features_for_clv = rfm_with_churn[['Recency', 'Frequency', 'MonetaryValue']].copy()
    for col in rfm_features_for_clv.columns:
         rfm_features_for_clv[col] = np.log1p(rfm_features_for_clv[col])
    rfm_with_churn['Predicted_CLV_90_Days'] = clv_model.predict(rfm_features_for_clv)

    print("\nSegmenting customers...")
    segmented_customers, segment_summary = customer_segmentation(rfm_with_churn)

    print("\n--- Customer Segments Summary ---")
    # Display summary with mapped names
    print(segmented_customers.groupby('Segment_Name')[['Recency', 'Frequency', 'MonetaryValue']].mean())


    print("\n--- Final Customer Data with Predictions and Segments ---")
    print(segmented_customers.head())
    
    # FIX: Reset index to turn 'customer_unique_id' into a column for plotting
    segmented_customers.reset_index(inplace=True)

    # --- Visualization ---
    fig_3d_segment = px.scatter_3d(segmented_customers,
                           x='Recency', y='Frequency', z='MonetaryValue',
                           color='Segment_Name',
                           title='Customer Segments in 3D RFM Space',
                           hover_data=['customer_unique_id', 'Predicted_CLV_90_Days'])
    fig_3d_segment.update_traces(marker=dict(size=3))
    fig_3d_segment.show()
    
    fig_3d_clv = px.scatter_3d(segmented_customers,
                           x='Recency', y='Frequency', z='MonetaryValue',
                           color='Predicted_CLV_90_Days',
                           color_continuous_scale='Viridis',
                           title='Customer Value Distribution in 3D RFM Space',
                           hover_data=['customer_unique_id', 'Segment_Name'])
    fig_3d_clv.update_traces(marker=dict(size=3))
    fig_3d_clv.show()

if __name__ == '__main__':
    main()

