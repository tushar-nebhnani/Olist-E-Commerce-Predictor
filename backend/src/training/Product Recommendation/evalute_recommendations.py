import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

DATA_PATH = r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\data\processed"
K = 10

def evaluate_svd_model():
    print("--- 1. Loading Data for Evaluation ---")
    orders_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_orders_cleaned_dataset.parquet'))
    order_items_df = pd.read_parquet(os.path.join(DATA_PATH, 'olist_order_items_cleaned_dataset.parquet'))
    
    df = pd.merge(orders_df, order_items_df, on='order_id')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    interactions_df = df[['customer_id', 'product_id', 'order_purchase_timestamp']].copy()
    interactions_df.sort_values(by='order_purchase_timestamp', inplace=True)

    print("\n--- 2. Creating a Time-Based Train-Test Split ---")
    split_point = int(len(interactions_df) * 0.80)
    train_df = interactions_df.iloc[:split_point]
    test_df = interactions_df.iloc[split_point:]
    
    print("\n--- 3. Training a TEMPORARY Model on the Training Set ---")
    user_cat = train_df['customer_id'].astype('category')
    product_cat = train_df['product_id'].astype('category')
    row_indices = user_cat.cat.codes
    col_indices = product_cat.cat.codes
    inv_user_map = {user_id: i for i, user_id in enumerate(user_cat.cat.categories)}
    product_map = {i: product_id for i, product_id in enumerate(product_cat.cat.categories)}

    train_user_item_sparse = csr_matrix((np.ones(len(train_df)), (row_indices, col_indices)),
                                         shape=(len(user_cat.cat.categories), len(product_cat.cat.categories)))

    svd = TruncatedSVD(n_components=100, random_state=42)
    svd.fit(train_user_item_sparse)

    print("\n--- 4. Evaluating Model Performance on REPEAT Customers ---")
    
    # Create the ground truth for all users in the test period
    ground_truth_all_users = test_df.groupby('customer_id')['product_id'].apply(set).to_dict()
    
    # THE FIX: Create a new ground truth dict containing only users that are ALSO in the training set.
    train_users_set = set(train_df['customer_id'])
    ground_truth_repeat_users = {user_id: items for user_id, items in ground_truth_all_users.items() if user_id in train_users_set}
    
    if not ground_truth_repeat_users:
        print("❌ Error: No repeat customers found in the test set. Cannot evaluate.")
        return

    print(f"✅ Found {len(ground_truth_repeat_users)} repeat customers to evaluate.")
    
    precisions = []
    recalls = []

    # Loop over the filtered dictionary of repeat users
    for user_id, actual_items in tqdm(ground_truth_repeat_users.items(), desc="Evaluating Repeat Customers"):
        user_index = inv_user_map.get(user_id)
        # This check is now mostly for safety; we know the user exists.
        if user_index is None:
            continue

        user_scores = np.dot(svd.transform(train_user_item_sparse[user_index]), svd.components_)
        items_bought_in_train = set(train_df[train_df['customer_id'] == user_id]['product_id'])
        
        recommendations = []
        for prod_index, score in enumerate(user_scores):
            prod_id = product_map.get(prod_index)
            if prod_id and prod_id not in items_bought_in_train:
                recommendations.append((prod_id, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        top_k_recs = {rec[0] for rec in recommendations[:K]}

        hits = len(top_k_recs.intersection(actual_items))
        precision = hits / K
        recall = hits / len(actual_items) if actual_items else 0
        precisions.append(precision)
        recalls.append(recall)

    print("\n--- 5. Final Evaluation Results ---")
    avg_precision = np.mean(precisions) * 100
    avg_recall = np.mean(recalls) * 100

    print(f"📊 Average Precision@{K}: {avg_precision:.2f}%")
    print(f"📊 Average Recall@{K}: {avg_recall:.2f}%")

if __name__ == "__main__":
    evaluate_svd_model()