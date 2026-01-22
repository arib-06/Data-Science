from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Load dataset (Iris)
data = datasets.load_iris()
X = data.data[:, :2]  # Use only first two features for visualization
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Kernels to compare
kernels = ['linear', 'poly', 'rbf']

print("SVM Kernel Decision Boundary Visualization on Iris Dataset")

# ---- PLOT: Decision Boundaries (Hyperplanes) ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, kernel in enumerate(kernels):
    svm = SVC(kernel=kernel, gamma='auto', random_state=42)
    svm.fit(X_train, y_train)

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot contour and training points
    axes[i].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
    axes[i].set_title(f"SVM with {kernel.upper()} kernel")
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')

plt.tight_layout()
plt.show()