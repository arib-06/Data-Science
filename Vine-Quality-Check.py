import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)


# 1. LOAD DATA

df = pd.read_csv("wine-quality-white-and-red.csv")

# ENCODE categorical column (THIS WAS THE ISSUE)
df['type'] = df['type'].map({'red': 0, 'white': 1})


# 2. BASIC CHECKS

print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nData types:\n", df.dtypes)


# 3. CREATE LABEL FOR CLASSIFICATION
df['label_goodwine'] = (df['quality'] >= 7).astype(int)

#  OPTION A: REGRESSION

X_reg = df.drop(['quality', 'label_goodwine'], axis=1)
y_reg = df['quality']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
Xr_train = scaler_reg.fit_transform(Xr_train)
Xr_test = scaler_reg.transform(Xr_test)

knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(Xr_train, yr_train)

yr_pred = knn_reg.predict(Xr_test)

rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
mae = mean_absolute_error(yr_test, yr_pred)
r2 = r2_score(yr_test, yr_pred)

print("\n===== KNN REGRESSION RESULTS =====")
print("RMSE:", rmse)
print("MAE :", mae)
print("R²  :", r2)


# Option B: Classification

X_cls = df.drop(['quality', 'label_goodwine'], axis=1)
y_cls = df['label_goodwine']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

scaler_cls = StandardScaler()
Xc_train = scaler_cls.fit_transform(Xc_train)
Xc_test = scaler_cls.transform(Xc_test)

knn_cls = KNeighborsClassifier(n_neighbors=5)
knn_cls.fit(Xc_train, yc_train)

yc_pred = knn_cls.predict(Xc_test)

cm = confusion_matrix(yc_test, yc_pred)

print("\n===== KNN CLASSIFICATION RESULTS =====")
print("Confusion Matrix:\n", cm)
print("Accuracy :", accuracy_score(yc_test, yc_pred))
print("Precision:", precision_score(yc_test, yc_pred))
print("Recall   :", recall_score(yc_test, yc_pred))
print("F1 Score :", f1_score(yc_test, yc_pred))

