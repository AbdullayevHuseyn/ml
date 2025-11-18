import numpy as np
from sklearn.linear_model import LinearRegression

# X must be 2D for scikit-learn: one column per feature
X = np.array([[90], [120], [150], [100], [130]])  # Ad spend
y = np.array([1000, 1300, 1800, 1200, 1380])      # Revenue

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Model parameters
print("Slope (m):", model.coef_)
print("Intercept (b):", model.intercept_)

# Predict new values
X_new = np.array([[200]])  # 2025 ad spends
y_pred = model.predict(X_new)
print("Predicted revenue:", y_pred.round(decimals=0))