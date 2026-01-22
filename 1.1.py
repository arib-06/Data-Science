import pandas as pd
data = {'Name': ['Alice', 'Bob'], 'Marks': [85, 90]}
df = pd.DataFrame(data)
print("Initial DataFrame:\n", df)
print("\nFiltered DataFrame (Marks > 85):")
print(df[df['Marks'] > 85])

import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y, label='Line')
plt.title('Sample Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression()
model.fit(X, y)
print("\nPrediction for input 5:", model.predict([[5]]))
