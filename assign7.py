# assign7.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- 1. Load dataset ----------
CSV_PATH = r"C:\Users\LEGION\Desktop\lab\ML\assignement 7\WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(CSV_PATH)

print("Data loaded. Shape:", df.shape)

# ---------- 2. Drop useless columns ----------
drop_cols = ['Attrition', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=c)

# ---------- 3. Encode categoricals & scale ----------
X = pd.get_dummies(df, drop_first=True)       # convert categories → numbers
X_scaled = StandardScaler().fit_transform(X)  # normalize features

print("Processed data shape:", X_scaled.shape)

# ---------- 4. Elbow Method ----------
wcss = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)   # inertia_ = WCSS

plt.plot(K, wcss, marker='o')
plt.title("Elbow Method (Employee Attrition Data)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ---------- 5. Choose optimal k (say k=3, check elbow curve) ----------
k_opt = 3
kmeans_opt = KMeans(n_clusters=k_opt, random_state=0, n_init=10)
labels = kmeans_opt.fit_predict(X_scaled)

print(f"\nKMeans clustering done with k={k_opt}")
print("Cluster centers (scaled features):\n", kmeans_opt.cluster_centers_)

# ---------- 6. Attach cluster labels back to data ----------
df_clusters = df.copy()
df_clusters['Cluster'] = labels
print("\nSample clustered data:")
print(df_clusters.head())
from sklearn.decomposition import PCA

# ---------- 7. Visualize clusters ----------
# Reduce dimensions to 2D for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7)

# Plot cluster centers (projected into PCA space)
centers_pca = pca.transform(kmeans_opt.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="red", s=200, marker="X", label="Centroids")

plt.title(f"KMeans Clustering (k={k_opt}) - PCA Projection")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()
