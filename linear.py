# Bestsellers Data Analysis
# My First Data Science Project 🚀

import pandas as pd
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score
import numpy as np
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("bestsellers with categories.csv")

# Clean the data
df.drop_duplicates(inplace=True)

df.rename(columns={
    'Name': 'Title',
    'Year': 'Publication Year',
    'User Rating': 'Ratings',
    'Reviews': 'Book Review',
    'Price': 'Book Price'
}, inplace=True)

df['Book Price'] = df['Book Price'].astype(float)

print("\nColumns after cleaning:", df.columns)

# -----------------------------
# Linear Regression (Regression)
# -----------------------------
print("\nLinear Regression (predicting Book Price)")

X = df[['Ratings', 'Book Review', 'Publication Year']]
y = df['Book Price']

lin_model = LinearRegression()
lin_model.fit(X, y)

y_pred = lin_model.predict(X)

print("R^2 Score:", r2_score(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

# Logistic Regression (Classification: Fiction vs Non Fiction)
print("\nLogistic Regression (predicting Genre: Fiction vs Non Fiction)")

# Features
X = df[['Ratings', 'Book Review', 'Publication Year']]

# Target (Genre → convert to 0/1)
y = df['Genre'].map({'Fiction': 0, 'Non Fiction': 1})

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=200 )
log_model.fit(X, y)

# Predictions
y_pred = log_model.predict(X)

# Accuracy
print("Accuracy:", accuracy_score(y, y_pred))




X = np.arange(1,10).reshape(-1,1)
y_reg = np.arange(2,11)       # continuous
y_cls = [0,0,0,0,1,1,1,1,1]   # binary

plt.subplot(1,2,1)
plt.scatter(X,y_reg); plt.plot(X,LinearRegression().fit(X,y_reg).predict(X))
plt.title("Linear Regression")

plt.subplot(1,2,2)
plt.scatter(X,y_cls); plt.plot(X,LogisticRegression().fit(X,y_cls).predict_proba(X)[:,1])
plt.title("Logistic Regression")

plt.show()


