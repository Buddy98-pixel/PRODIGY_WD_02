import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- STEP 1: GENERATE CUSTOMER DATA ---
np.random.seed(42)
n_customers = 300

# Creating 3 distinct types of customers (Clusters)
# Cluster 1: Low Income, Low Spend
# Cluster 2: High Income, High Spend
# Cluster 3: Mid Income, Mid Spend
income = np.concatenate([
    np.random.normal(25, 5, 100), 
    np.random.normal(80, 10, 100), 
    np.random.normal(50, 8, 100)
])
spending = np.concatenate([
    np.random.normal(20, 5, 100), 
    np.random.normal(85, 10, 100), 
    np.random.normal(50, 8, 100)
])

df = pd.DataFrame({'AnnualIncome': income, 'SpendingScore': spending})

# --- STEP 2: PRE-PROCESSING ---
# K-Means is based on distance (Euclidean distance). 
# If one column is 0-100 and another is 0-100,000, the model fails.
# We "scale" the data so everything is on a similar level.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# --- STEP 3: FIND THE OPTIMAL 'K' (The Elbow Method) ---
# How many groups do we need? 2? 5? 10?
# We run the model multiple times and look for the "elbow."
wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# --- STEP 4: TRAIN THE FINAL MODEL ---
# Let's assume 3 clusters based on our data generation
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# --- STEP 5: VISUALIZE THE GROUPS ---
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['AnnualIncome'], cluster_data['SpendingScore'], 
                label=f'Cluster {i}', c=colors[i], edgecolors='black')

# Plot the center of each cluster (Centroids)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', marker='*', label='Centroids')

plt.title('Customer Segments (K-Means)')
plt.xlabel('Annual Income ($k)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()