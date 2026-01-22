import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler

X1, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

np.random.seed(42)
noise = np.random.uniform(low=-2, high=2, size=(20, 2))

X = np.vstack([X1, noise])

X = StandardScaler().fit_transform(X)


dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)

# Analyze results
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("=" * 60)
print("DBSCAN CLUSTERING ANALYSIS")
print("=" * 60)
print(f"\nDataset: {len(X)} total points")
print(f"\nParameters:")
print(f"  eps (neighborhood radius): {0.3}")
print(f"  min_samples: {5}")
print(f"\nResults:")
print(f"  Number of clusters found: {n_clusters}")
print(f"  Number of noise points: {n_noise}")
print(f"  Percentage of noise: {(n_noise/len(X))*100:.2f}%")

# Show cluster sizes
print(f"\nCluster Sizes:")
for i in range(n_clusters):
    cluster_size = list(labels).count(i)
    print(f"  Cluster {i}: {cluster_size} points")

# Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Original Data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', s=50, alpha=0.6)
plt.title('Original Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)

# Plot 2: DBSCAN Results
plt.subplot(1, 2, 2)
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points in black
        color = 'black'
        marker = 'x'
        label_name = 'Noise'
    else:
        marker = 'o'
        label_name = f'Cluster {label}'
    
    mask = labels == label
    plt.scatter(X[mask, 0], X[mask, 1], c=[color], label=label_name,
                s=50, alpha=0.6, marker=marker)

plt.title('DBSCAN Clustering Results', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

