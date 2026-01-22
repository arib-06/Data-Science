import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

# Sample dataset
data = pd.DataFrame({
    "Feature1": [10, 20, 30, 40, 50, 100],
    "Feature2": [100, 200, 300, 400, 500, 1000]
})

# Apply different scaling techniques
scalers = {
    "Min-Max": MinMaxScaler(),
    "Standardization": StandardScaler(),
    "Robust": RobustScaler(),
    "MaxAbs": MaxAbsScaler()
}

scaled_data = {}
for name, scaler in scalers.items():
    scaled = scaler.fit_transform(data)
    scaled_data[name] = pd.DataFrame(scaled, columns=data.columns)

# Plot both features
fig, axes = plt.subplots(2, 5, figsize=(18, 8))  # 2 rows 

# --- Original Features ---
axes[0, 0].hist(data['Feature1'], bins=10, color="skyblue")
axes[0, 0].set_title("Original Feature1")

axes[1, 0].hist(data['Feature2'], bins=10, color="skyblue")
axes[1, 0].set_title("Original Feature2")

# --- Scaled Features ---
i = 1
for name, df in scaled_data.items():
    # Feature1
    axes[0, i].hist(df['Feature1'], bins=10, color="orange")
    axes[0, i].set_title(f"{name} Feature1")

    # Feature2
    axes[1, i].hist(df['Feature2'], bins=10, color="orange")
    axes[1, i].set_title(f"{name} Feature2")

    i += 1

plt.tight_layout()
plt.show()
