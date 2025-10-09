import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

def load_data(path_dict):
    """Loads all necessary parquet files into pandas DataFrames."""
    data = {}
    for key, path in path_dict.items():
        try:
            data[key] = pd.read_parquet(path)
        except FileNotFoundError:
            print(f"Error: The file at {path} was not found.")
            return None
    return data

def preprocess_data(data):
    """Merges and preprocesses the Olist dataset."""
    orders_customers = pd.merge(data['orders'], data['customers'], on='customer_id')
    orders_payments = pd.merge(orders_customers, data['payments'], on='order_id')
    order_items_products = pd.merge(data['order_items'], data['products'], on='product_id')
    full_data = pd.merge(orders_payments, order_items_products, on='order_id')

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

def feature_engineering(df, rfm, snapshot_date):
    """Creates additional features to improve model performance."""
    # Calculate Average Order Value
    rfm['Average_Order_Value'] = rfm['MonetaryValue'] / rfm['Frequency']
    
    # Calculate Days Since First Purchase
    first_purchase_date = df.groupby('customer_unique_id')['order_purchase_timestamp'].min()
    rfm['Days_Since_First_Purchase'] = (snapshot_date - first_purchase_date).dt.days
    
    # Fill any potential NaNs
    rfm.fillna(0, inplace=True)
    return rfm

def create_churn_feature(rfm):
    """Creates the target variable 'Churn' based on recency."""
    churn_threshold = rfm['Recency'].quantile(0.75)
    rfm['Churn'] = (rfm['Recency'] > churn_threshold).astype(int)
    return rfm

def train_churn_model(rfm_df):
    """Trains a LightGBM classifier to predict customer churn using SMOTE."""
    print("--- Training Churn Model ---")
    
    # Use all engineered features except Recency to avoid data leakage
    features = ['Frequency', 'MonetaryValue', 'Average_Order_Value', 'Days_Since_First_Purchase']
    X = rfm_df[features]
    y = rfm_df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Apply SMOTE to the training data to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Use LightGBM for better performance
    model = LGBMClassifier(random_state=42, objective='binary', is_unbalance=True)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)

    print("\n--- Churn Model Performance (Improved) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    rfm_df['Churn_Probability'] = model.predict_proba(X)[:, 1]
    return model, rfm_df

def train_clv_model(df):
    """Prepares data and trains a two-part model to predict Customer Lifetime Value."""
    print("\n--- Training CLV Two-Part Model ---")
    
    df = df.sort_values('order_purchase_timestamp')
    split_point = df['order_purchase_timestamp'].max() - pd.DateOffset(days=90)
    
    calibration_df = df[df['order_purchase_timestamp'] < split_point]
    observation_df = df[df['order_purchase_timestamp'] >= split_point]

    calibration_rfm = calculate_rfm(calibration_df, split_point)
    calibration_rfm_featured = feature_engineering(calibration_df, calibration_rfm, split_point)
    
    observation_value = observation_df.groupby('customer_unique_id')['price'].sum().rename('Future_CLV')
    
    clv_df = pd.merge(calibration_rfm_featured, observation_value, on='customer_unique_id', how='left')
    clv_df['Future_CLV'].fillna(0, inplace=True)

    features = ['Recency', 'Frequency', 'MonetaryValue', 'Average_Order_Value', 'Days_Since_First_Purchase']
    X = clv_df[features]
    
    # --- Part 1: Probability Model (Will the customer buy?) ---
    clv_df['Will_Buy'] = (clv_df['Future_CLV'] > 0).astype(int)
    y_class = clv_df['Will_Buy']
    
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.3, random_state=42, stratify=y_class)
    
    # FIX: Add scale_pos_weight for imbalance and tune hyperparameters
    neg_count = y_train_class.value_counts()[0]
    pos_count = y_train_class.value_counts()[1]
    scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1

    prob_model = LGBMClassifier(random_state=42, objective='binary', scale_pos_weight=scale_pos_weight_value)
    prob_model.fit(X_train, y_train_class)
    print("\nCLV - Probability Model Trained.")
    
    # Print performance of the probability model
    y_pred_class = prob_model.predict(X_test)
    print("\n--- CLV Probability Model Performance ---")
    print(classification_report(y_test_class, y_pred_class))


    # --- Part 2: Spend Model (If they buy, how much?) ---
    spenders_df = clv_df[clv_df['Will_Buy'] == 1]
    X_spend = spenders_df[features]
    y_spend = spenders_df['Future_CLV']
    
    for col in features:
        X_spend[col] = np.log1p(X_spend[col])
    y_spend_log = np.log1p(y_spend)

    X_train_spend, X_test_spend, y_train_spend_log, y_test_spend_log = train_test_split(X_spend, y_spend_log, test_size=0.3, random_state=42)

    # FIX: Add hyperparameters to the regression model to prevent warnings and improve generalization
    spend_model = LGBMRegressor(
        random_state=42, 
        objective='regression_l1',
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=20,
        min_child_samples=5,
        n_jobs=-1
    )
    spend_model.fit(X_train_spend, y_train_spend_log)
    print("CLV - Spend Model Trained.")
    
    # --- Evaluate the combined model ---
    buy_prob = prob_model.predict_proba(X_test)[:, 1]
    
    X_test_spend_eval = X_test.copy()
    for col in features:
        X_test_spend_eval[col] = np.log1p(X_test_spend_eval[col])
    predicted_spend_log = spend_model.predict(X_test_spend_eval)
    predicted_spend = np.expm1(predicted_spend_log)

    y_pred_combined = buy_prob * predicted_spend
    y_test_combined = clv_df.loc[y_test_class.index, 'Future_CLV']

    mae = mean_absolute_error(y_test_combined, y_pred_combined)
    r2 = r2_score(y_test_combined, y_pred_combined)

    print("\n--- CLV Two-Part Model Performance (Improved) ---")
    print(f"CLV Model Mean Absolute Error (MAE): {mae:.2f}")
    print(f"CLV Model R-squared (R²): {r2:.2f}")

    return prob_model, spend_model

def customer_segmentation(rfm_df):
    """Segments customers using K-Means clustering on RFM values."""
    rfm_scaled = StandardScaler().fit_transform(rfm_df[['Recency', 'Frequency', 'MonetaryValue']])
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    rfm_df['Segment'] = kmeans.fit_predict(rfm_scaled)

    segment_stats = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'MonetaryValue']].mean()
    
    active_segment = segment_stats.sort_values('Recency', ascending=True).index[0]
    
    segment_map = {
        active_segment: 'Active Customers',
        1 - active_segment: 'At Risk / Churned'
    }
            
    rfm_df['Segment_Name'] = rfm_df['Segment'].map(segment_map)
    return rfm_df, segment_stats

def main():
    """Main function to run the customer analytics pipeline."""
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
        return

    print("Preprocessing data...")
    processed_data = preprocess_data(olist_data)

    snapshot_date_full = processed_data['order_purchase_timestamp'].max() + pd.DateOffset(days=1)
    
    print("Calculating RFM and engineering features...")
    rfm_full_data = calculate_rfm(processed_data, snapshot_date_full)
    featured_rfm = feature_engineering(processed_data, rfm_full_data, snapshot_date_full)

    print("Creating churn feature...")
    rfm_with_churn = create_churn_feature(featured_rfm)

    churn_model, rfm_with_churn = train_churn_model(rfm_with_churn)
    clv_prob_model, clv_spend_model = train_clv_model(processed_data)

    # Predict CLV for all customers using the two-part model
    features_for_clv = rfm_with_churn[['Recency', 'Frequency', 'MonetaryValue', 'Average_Order_Value', 'Days_Since_First_Purchase']].copy()
    
    # Predict probability
    buy_probability = clv_prob_model.predict_proba(features_for_clv)[:, 1]
    
    # Predict spend (after log transforming features)
    features_for_spend_pred = features_for_clv.copy()
    for col in features_for_spend_pred.columns:
         features_for_spend_pred[col] = np.log1p(features_for_spend_pred[col])
    
    predicted_spend_log = clv_spend_model.predict(features_for_spend_pred)
    predicted_spend = np.expm1(predicted_spend_log)

    # Combine predictions
    rfm_with_churn['Predicted_CLV_90_Days'] = buy_probability * predicted_spend


    print("\nSegmenting customers...")
    segmented_customers, _ = customer_segmentation(rfm_with_churn)

    print("\n--- Customer Segments Summary ---")
    print(segmented_customers.groupby('Segment_Name')[['Recency', 'Frequency', 'MonetaryValue']].mean())

    print("\n--- Final Customer Data with Predictions and Segments ---")
    print(segmented_customers.head())
    
    # --- Save Models ---
    print("\nSaving trained models to disk...")
    # Define the save path from your project structure
    save_dir = r'D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Churn Prediction'
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump(churn_model, os.path.join(save_dir, 'churn_model.joblib'))
    joblib.dump(clv_prob_model, os.path.join(save_dir, 'clv_probability_model.joblib'))
    joblib.dump(clv_spend_model, os.path.join(save_dir, 'clv_spend_model.joblib'))
    print(f"Models saved successfully in {save_dir}.")

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

