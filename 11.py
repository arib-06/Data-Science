import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_iris(return_X_y=True)
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=4).fit(Xs)
X2 = pca.transform(Xs)[:, :2]

print("Explained variance ratios:", np.round(pca.explained_variance_ratio_, 4),
      "\nCum:", np.round(pca.explained_variance_ratio_.cumsum(), 4))

# 2D scatter of first two PCs
names = load_iris().target_names
markers = ['o','s','^']

plt.figure(figsize=(6,5))
for i, lab in enumerate(np.unique(y)):
    pts = X2[y==lab]
    plt.scatter(pts[:,0], pts[:,1], label=names[lab], marker=markers[i],
                edgecolor='k', s=50)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend()
plt.grid(True)
plt.title("PCA (Iris)")
plt.tight_layout()
plt.show()

# Scree + cumulative plot
plt.figure(figsize=(6,3))
r = pca.explained_variance_ratio_
plt.bar(range(1,len(r)+1), r, alpha=0.6)
plt.plot(range(1,len(r)+1), r.cumsum(), marker='o')
plt.xlabel("PC")
plt.ylabel("Explained variance")
plt.tight_layout()
plt.s
