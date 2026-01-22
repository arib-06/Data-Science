# Assignment6.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Loading file
CSV_PATH = r"C:\Users\LEGION\Desktop\lab\ML\assignement 6\WA_Fn-UseC_-HR-Employee-Attrition.csv"

LR = 0.5
EPOCHS = 3000
TEST_SIZE = 0.2
RANDOM_STATE = 42
PRINT_EVERY = 500  # show loss every N epochs (set 0 to disable)

# Load & preprocess
df = pd.read_csv(CSV_PATH)
df['target'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# drop identifier/constant columns if present
for c in ('EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours', 'Attrition'):
    if c in df.columns:
        df = df.drop(columns=c)

# one-hot encode categorical columns
X_df = pd.get_dummies(df.drop(columns='target'), drop_first=True)

# fill missing values
if X_df.isnull().sum().sum() > 0:
    X_df = X_df.fillna(X_df.mean())

# scale numeric columns
num_cols = X_df.select_dtypes(include=[np.number]).columns
if len(num_cols) > 0:
    X_df[num_cols] = StandardScaler().fit_transform(X_df[num_cols])

# convert to numpy arrays
X = X_df.to_numpy(dtype=np.float64)
y = df['target'].to_numpy(dtype=np.int64)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

#  Logistic Regression (Gradient Descent)
def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic_gd(X, y, lr=0.1, epochs=1000, verbose_every=0):
    m, n = X.shape
    w = np.zeros(n, dtype=np.float64)
    b = 0.0
    for epoch in range(1, epochs + 1):
        z = X.dot(w) + b
        p = sigmoid(z)

        dw = (1.0 / m) * X.T.dot(p - y)
        db = (1.0 / m) * (p - y).sum()

        w -= lr * dw
        b -= lr * db

        if verbose_every and (epoch % verbose_every == 0 or epoch == epochs):
            loss = - (1.0/m) * np.sum(
                y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15)
            )
            print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")
    return w, b

def predict(X, w, b, threshold=0.5):
    return (sigmoid(X.dot(w) + b) >= threshold).astype(int)

#  Train custom GD model
w, b = train_logistic_gd(X_train, y_train, lr=LR, epochs=EPOCHS, verbose_every=PRINT_EVERY)
y_pred_custom = predict(X_test, w, b)

print("\nCustom Logistic Regression (GD):")
print("Accuracy:", accuracy_score(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom, zero_division=0))

# Compare with sklearn LogisticRegression 
sk = LogisticRegression(max_iter=1000, solver='lbfgs')
sk.fit(X_train, y_train)
y_pred_sk = sk.predict(X_test)

print("sklearn LogisticRegression:")
print("Accuracy:", accuracy_score(y_test, y_pred_sk))
print(classification_report(y_test, y_pred_sk, zero_division=0))

