import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- Data Loading and Preparation ---
datas = pd.read_csv('Position_Salaries.csv')

# Use .values for numpy array conversion (standard practice for scikit-learn input)
X = datas.iloc[:, 1:2].values  # Level (features)
y = datas.iloc[:, 2].values    # Salary (target)

# --- Polynomial Regression Model Training (Degree 4) ---
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

# Note: poly.fit(X_poly, y) is redundant here as X_poly is already fit_transformed
pol = LinearRegression()
pol.fit(X_poly, y)

# --- Visualization for Smoother Curve ---
# Create a dense range of levels (e.g., from min to max) for a smoother plot
X_grid = np.arange(min(X), max(X), 0.1)  # Generates 10 points between each original point
X_grid = X_grid.reshape((len(X_grid), 1))

plt.figure(figsize=(10, 6)) # Add a figure size for better viewing

plt.scatter(X, y, color='blue', label='Actual Data Points')

# Plot the smoother, predicted curve
plt.plot(X_grid, pol.predict(poly.fit_transform(X_grid)),
         color='red', label='Polynomial Fit (Degree 4)')

plt.title('Polynomial Regression: Salary vs. Level')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.grid(True) # Add a grid for better readability
plt.show()

# --- Prediction for a New Result (Level 11) ---
new_level = 11
new_level_array = np.array([[new_level]])

# Predict a new result with polynomial regression
# Assign the prediction to a variable to control output
predicted_salary = pol.predict(poly.fit_transform(new_level_array))

print(f'\nPrediction for Level {new_level}:')
print(f'Polynomial Regression result: ${predicted_salary[0]:,.2f}') # Format as currency

