import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb
from haversine import haversine, Unit
from pathlib import Path
import logging
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION AND SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_PATH = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
OUTPUTS_PATH = PROJECT_ROOT / "models" / "purchase_prediction"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

NEGATIVE_SAMPLE_RATIO = 4
RANDOM_STATE = 42

# --- 2. DATA LOADING & PREPARATION FUNCTIONS ---

def load_individual_datasets(path: Path) -> dict:
    """Loads all individual cleaned parquet files."""
    logging.info(f"Loading individual cleaned datasets from: '{path}'...")
    datasets = {}
    required_files = {
        'orders': 'olist_orders_cleaned_dataset', 'customers': 'olist_customers_cleaned_dataset',
        'reviews': 'olist_order_reviews_cleaned_dataset', 'payments': 'olist_order_payments_cleaned_dataset',
        'items': 'olist_order_items_cleaned_dataset', 'products': 'olist_products_cleaned_dataset',
        'sellers': 'olist_sellers_cleaned_dataset', 'translation': 'category_name_translation_cleaned_dataset',
        'geolocation': 'olist_geolocation_cleaned_dataset'
    }
    for key, filename in required_files.items():
        datasets[key] = pd.read_parquet(path / f"{filename}.parquet")
    return datasets

def create_master_table(datasets: dict) -> pd.DataFrame:
    """Merges individual dataframes into a single master table."""
    logging.info("Creating master dataset...")
    payments_agg = datasets['payments'].groupby('order_id').agg(
        total_payment_value=('payment_value', 'sum'),
    ).reset_index()

    geo = datasets['geolocation'].groupby('geolocation_zip_code_prefix').agg(
        geolocation_lat=('geolocation_lat', 'mean'),
        geolocation_lng=('geolocation_lng', 'mean')
    ).reset_index()

    df = pd.merge(datasets['orders'], datasets['customers'], on='customer_id', how='left')
    df = pd.merge(df, datasets['reviews'], on='order_id', how='left')
    df = pd.merge(df, payments_agg, on='order_id', how='left')
    df = pd.merge(df, datasets['items'], on='order_id', how='left')
    df = pd.merge(df, datasets['products'], on='product_id', how='left')
    df = pd.merge(df, datasets['sellers'], on='seller_id', how='left')
    df = pd.merge(df, datasets['translation'], on='product_category_name', how='left')
    
    df = pd.merge(df, geo, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}, inplace=True)
    df = pd.merge(df, geo, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left', suffixes=('_customer', '_seller'))
    df.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
    
    df.drop(['geolocation_zip_code_prefix_customer', 'geolocation_zip_code_prefix_seller'], axis=1, inplace=True)
    df.dropna(subset=['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng', 'product_id', 'customer_unique_id'], inplace=True)
    
    return df

def engineer_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates advanced behavioral and reputational features."""
    logging.info("Engineering advanced behavioral and reputational features...")

    # Customer-level features
    customer_stats = df.groupby('customer_unique_id').agg(
        customer_avg_review_score=('review_score', 'mean'),
        customer_order_count=('order_id', 'nunique'),
        customer_total_spend=('total_payment_value', 'sum')
    ).reset_index()

    # Product-level features
    product_stats = df.groupby('product_id').agg(
        product_popularity=('order_id', 'nunique'),
        product_avg_review_score=('review_score', 'mean'),
        product_avg_price=('price', 'mean')
    ).reset_index()

    # Category-level features
    category_stats = df.groupby('product_category_name_english').agg(
        category_popularity=('order_id', 'nunique'),
        category_avg_review_score=('review_score', 'mean')
    ).reset_index()

    # Merge new features back into the main dataframe
    df = pd.merge(df, customer_stats, on='customer_unique_id', how='left')
    df = pd.merge(df, product_stats, on='product_id', how='left')
    df = pd.merge(df, category_stats, on='product_category_name_english', how='left')
    
    logging.info("  -> Advanced features created successfully.")
    return df

def engineer_interaction_features_and_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Performs interaction feature engineering and negative sampling."""
    logging.info("Starting interaction feature engineering and negative sampling...")

    df['distance_km'] = df.apply(lambda row: haversine((row['customer_lat'], row['customer_lng']), (row['seller_lat'], row['seller_lng']), unit=Unit.KILOMETERS), axis=1)
    df['price_vs_category_avg'] = df['price'] - df['product_avg_price']

    positive_samples = df[['customer_unique_id', 'product_id']].copy()
    positive_samples.drop_duplicates(inplace=True)
    positive_samples['purchased'] = 1

    n_pos_samples = len(positive_samples)
    n_neg_samples_to_generate = n_pos_samples * (NEGATIVE_SAMPLE_RATIO + 5)
    
    random_customers = np.random.choice(df['customer_unique_id'].unique(), size=n_neg_samples_to_generate)
    random_products = np.random.choice(df['product_id'].unique(), size=n_neg_samples_to_generate)
    
    negative_samples = pd.DataFrame({'customer_unique_id': random_customers, 'product_id': random_products})
    negative_samples.drop_duplicates(inplace=True)
    
    merged = pd.merge(negative_samples, positive_samples, on=['customer_unique_id', 'product_id'], how='left', indicator=True)
    pure_negatives = merged[merged['_merge'] == 'left_only'][['customer_unique_id', 'product_id']]
    
    n_neg_samples_needed = n_pos_samples * NEGATIVE_SAMPLE_RATIO
    sampled_negatives = pure_negatives.sample(n=min(n_neg_samples_needed, len(pure_negatives)), random_state=RANDOM_STATE)
    sampled_negatives['purchased'] = 0

    model_df = pd.concat([positive_samples, sampled_negatives], ignore_index=True)

    # Merge all features into the final modeling dataframe
    feature_columns = [
        'customer_unique_id', 'customer_state', 'customer_avg_review_score', 'customer_order_count', 'customer_total_spend',
        'product_id', 'product_category_name_english', 'price', 'freight_value', 'product_photos_qty',
        'product_weight_g', 'product_volume_cm3', 'seller_state', 'distance_km', 'price_vs_category_avg',
        'product_popularity', 'product_avg_review_score', 'category_popularity', 'category_avg_review_score'
    ]
    features_df = df.drop_duplicates(subset=['customer_unique_id', 'product_id'])
    
    # Select only necessary feature columns before merging to save memory
    customer_centric_features = features_df.drop_duplicates(subset=['customer_unique_id'])[['customer_unique_id', 'customer_state', 'customer_avg_review_score', 'customer_order_count', 'customer_total_spend']]
    product_centric_features = features_df.drop_duplicates(subset=['product_id'])[['product_id', 'product_category_name_english', 'price', 'freight_value', 'product_photos_qty', 'product_weight_g', 'product_volume_cm3', 'seller_state', 'distance_km', 'price_vs_category_avg', 'product_popularity', 'product_avg_review_score', 'category_popularity', 'category_avg_review_score']]

    model_df = pd.merge(model_df, customer_centric_features, on='customer_unique_id', how='left')
    model_df = pd.merge(model_df, product_centric_features, on='product_id', how='left')

    model_df.dropna(inplace=True)
    model_df = model_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    logging.info(f"  -> Final modeling dataset shape: {model_df.shape}")
    return model_df

# --- 3. MODEL TRAINING & SAVING FUNCTIONS ---

def train_and_save_pipeline(model_df: pd.DataFrame) -> None:
    """Trains and saves the v2 prediction model pipeline."""
    logging.info("Defining feature transformers and v2 model pipeline...")
    
    numerical_features = [
        'price', 'freight_value', 'product_photos_qty', 'product_weight_g',
        'product_volume_cm3', 'distance_km', 'customer_avg_review_score',
        'customer_order_count', 'customer_total_spend', 'product_popularity',
        'product_avg_review_score', 'price_vs_category_avg', 'category_popularity',
        'category_avg_review_score'
    ]
    categorical_features = [
        'product_category_name_english', 'customer_state', 'seller_state'
    ]

    X = model_df[numerical_features + categorical_features]
    y = model_df['purchased']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    
    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    logging.info(f"Calculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='drop')

    lgbm = lgb.LGBMClassifier(
        objective='binary', metric='auc', n_estimators=1000, learning_rate=0.05,
        num_leaves=40, max_depth=-1, min_child_samples=30, subsample=0.85,
        colsample_bytree=0.75, random_state=RANDOM_STATE, n_jobs=-1,
        scale_pos_weight=scale_pos_weight  # Key parameter to address imbalance
    )

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lgbm)])
    
    logging.info("Training the v2 LightGBM model...")
    
    preprocessor.fit(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    model_pipeline.fit(
        X_train, y_train,
        classifier__eval_set=[(X_test_transformed, y_test)],
        classifier__callbacks=[lgb.early_stopping(50, verbose=True)]
    )
    
    logging.info("Evaluating v2 model performance...")
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"Divine Accuracy v2 (Test AUC-ROC Score): {auc:.4f}")
    
    y_pred = model_pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info(f"Classification Report v2:\n{report}")
    
    report_file = OUTPUTS_PATH / "classification_report_v2.txt"
    with open(report_file, 'w') as f:
        f.write(f"AUC Score: {auc:.4f}\n\n")
        f.write(report)
    logging.info(f"Classification report saved to '{report_file}'")

    model_file = OUTPUTS_PATH / "purchase_prediction_pipeline_v2.joblib"
    joblib.dump(model_pipeline, model_file)
    logging.info(f"Trained pipeline v2 saved to '{model_file}'")

    # Feature Importance Plot
    feature_importances = model_pipeline.named_steps['classifier'].feature_importances_
    ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(ohe_feature_names)
    
    importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': feature_importances}).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 20 Feature Importances for v2 Model')
    plt.tight_layout()
    plt.savefig(OUTPUTS_PATH / 'feature_importance_v2.png')
    logging.info(f"Feature importance plot saved to '{OUTPUTS_PATH / 'feature_importance_v2.png'}'")

# --- 4. MAIN ORCHESTRATOR ---
def main_pipeline():
    """Main orchestrator for the v2 purchase prediction model pipeline."""
    logging.info("--- Starting God-Tier Purchase Prediction Pipeline v2 ---")
    
    datasets = load_individual_datasets(PROCESSED_DATA_PATH)
    master_df = create_master_table(datasets)
    master_df_advanced_features = engineer_advanced_features(master_df)
    modeling_data = engineer_interaction_features_and_sample(master_df_advanced_features)
    train_and_save_pipeline(modeling_data)
    
    logging.info("--- v2 Pipeline Finished Successfully ---")

if __name__ == '__main__':
    main_pipeline()
