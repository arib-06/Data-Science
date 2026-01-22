import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1️ Create sample data
df = pd.DataFrame({     
    'Age': [25, np.nan, 28, 35, np.nan, 40],
    'Salary': [50000, 54000, np.nan, 58000, 62000, np.nan],
    'Department': ['HR', 'IT', np.nan, 'Finance', 'IT', 'Finance']
})

print("Original Data:\n", df)

# 2️ Show missing values
print("\nMissing values per column:\n", df.isnull().sum())
sns.heatmap(df.isnull(), cbar=True, cmap='magma') 
plt.show()

# 3️ Fill missing numbers with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# 4️ Fill missing text with most common value
df['Department'] = df['Department'].fillna(df['Department'].mode()[0]) 

# 5️ Encode text columns
# One-Hot Encoding
onehot = pd.get_dummies(df['Department'], drop_first=True)   
print("\nOne-Hot Encoded Columns:\n", onehot)

# Label Encoding
df['Department_Label'] = LabelEncoder().fit_transform(df['Department'])  
print("\nLabel Encoded Column:\n", df[['Department', 'Department_Label']])

# 6️ Combine into final dataset
final_df = pd.concat([df, onehot], axis=1)

# Show all columns
pd.set_option('display.max_columns', None)
print("\nFinal Processed Data:\n", final_df)
