import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load Data
df = pd.read_csv('ecommerce_customer_behavior_dataset_v2.csv')

# 2. Select Features for Segmentation
# We focus on Spending, Volume, Discounts, Engagement, and Satisfaction
features = [
    'Total_Amount', 
    'Quantity', 
    'Discount_Amount', 
    'Session_Duration_Minutes', 
    'Pages_Viewed', 
    'Customer_Rating'
]

X = df[features].copy()

# 3. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Model (4 Clusters)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Analyze & Name the Clusters
# We calculate the mean of each feature for each cluster to identify them
means = df.groupby('Cluster')[features].mean()

# logic to map Cluster ID -> Persona Name dynamically
# This prevents labels from swapping if you retrain
cluster_mapping = {}

# Find "The Big Spender" (Max Total Amount)
big_spender_idx = means['Total_Amount'].idxmax()
cluster_mapping[big_spender_idx] = {
    'name': '💎 The Big Spender',
    'desc': 'High spender, buys in bulk. Uses discounts.',
    'action': 'Assign VIP account manager. Offer exclusive presales.'
}

# Find "The At-Risk Customer" (Lowest Rating)
at_risk_idx = means['Customer_Rating'].idxmin()
cluster_mapping[at_risk_idx] = {
    'name': '📉 The At-Risk Customer',
    'desc': 'Low satisfaction, low spending. Likely to churn.',
    'action': 'Send personalized apology or feedback survey.'
}

# Find "The Loyal Regular" (High Quantity, High Rating, but not the Whale)
# We exclude the ones we already found
remaining = [c for c in range(k) if c not in [big_spender_idx, at_risk_idx]]
# Of the remaining, the one with higher Quantity is the Regular
loyal_idx = means.loc[remaining, 'Quantity'].idxmax()
cluster_mapping[loyal_idx] = {
    'name': '💖 The Loyal Regular',
    'desc': 'Happy, frequent buyer with moderate spend.',
    'action': 'Enroll in Loyalty Program. Offer "Refer a Friend" bonus.'
}

# The last one is "The Thrifty Browser"
thrifty_idx = [c for c in range(k) if c not in cluster_mapping][0]
cluster_mapping[thrifty_idx] = {
    'name': '🛒 The Thrifty Browser',
    'desc': 'Browses often, buys little, but is generally happy.',
    'action': 'Nudge with "Free Shipping" or low-barrier offers.'
}

print("Cluster Identification:")
for i in range(k):
    print(f"Cluster {i}: {cluster_mapping[i]['name']}")

# 6. Save Artifacts
artifacts = {
    'scaler': scaler,
    'kmeans_model': kmeans,
    'features': features,
    'cluster_mapping': cluster_mapping,
    'means': means # Saved for plotting in the app
}
joblib.dump(artifacts, 'segmentation_artifacts.joblib')
print("\nSaved artifacts to 'segmentation_artifacts.joblib'")