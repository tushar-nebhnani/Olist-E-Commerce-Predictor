# check_valid_users.py
import joblib
import os

# Make sure this path points to your actual artifacts directory
ARTIFACTS_PATH = r"D:\Data Science\CaseStudy ML\Olist-E-Commerce-Predictor-\backend\models\Product Recommendation"

try:
    user_map = joblib.load(os.path.join(ARTIFACTS_PATH, 'svd_user_map.joblib'))
    
    # The user IDs are the values in the map dictionary
    valid_customer_ids = list(user_map.values())
    
    print(f"✅ Found {len(valid_customer_ids)} customers that the model was trained on.")
    print("\nHere are 5 valid customer IDs you can use for testing:")
    for i in range(5):
        print(f"  - {valid_customer_ids[i]}")

except FileNotFoundError:
    print(f"❌ Error: Could not find 'svd_user_map.joblib' at the specified path.")