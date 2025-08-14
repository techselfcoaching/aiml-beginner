import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("EXAMPLE 1: SIMPLE LINEAR REGRESSION (ONE VARIABLE)")
print("="*60)

# Generate sample data - House size vs Price
house_sizes = np.random.uniform(800, 3000, 100)  # Square feet
# Price = 50 * size + 20000 + some noise
house_prices = 50 * house_sizes + 20000 + np.random.normal(0, 5000, 100)

# Reshape for sklearn (needs 2D array)
X_single = house_sizes.reshape(-1, 1)
y_single = house_prices

# Create and fit the model
model_single = LinearRegression()
model_single.fit(X_single, y_single)

# Make predictions
y_pred_single = model_single.predict(X_single)

# Print results
print(f"Intercept (β₀): ${model_single.intercept_:,.2f}")
print(f"Slope (β₁): ${model_single.coef_[0]:.2f} per sq ft")
print(f"R² Score: {r2_score(y_single, y_pred_single):.3f}")
print(f"\nInterpretation:")
print(f"- When house size = 0 sq ft, predicted price = ${model_single.intercept_:,.2f}")
print(f"- For every 1 sq ft increase, price increases by ${model_single.coef_[0]:.2f}")

# Create visualization
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(house_sizes, house_prices, alpha=0.6, color='blue', label='Actual prices')
plt.plot(house_sizes, y_pred_single, color='red', linewidth=2, label='Regression line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Simple Linear Regression\nHouse Size vs Price')
plt.legend()
plt.grid(True, alpha=0.3)

print("\n" + "="*60)
print("EXAMPLE 2: MULTIPLE LINEAR REGRESSION (TWO VARIABLES)")
print("="*60)

# Generate sample data - House size and bedrooms vs Price
n_samples = 100
house_sizes_multi = np.random.uniform(800, 3000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
# Price = 40 * size + 5000 * bedrooms + 10000 + noise
house_prices_multi = (40 * house_sizes_multi + 
                     5000 * bedrooms + 
                     10000 + 
                     np.random.normal(0, 8000, n_samples))

# Prepare data
X_multi = np.column_stack([house_sizes_multi, bedrooms])
y_multi = house_prices_multi

# Create and fit the model
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

# Make predictions
y_pred_multi = model_multi.predict(X_multi)

# Print results
print(f"Intercept (β₀): ${model_multi.intercept_:,.2f}")
print(f"Coefficient for Size (β₁): ${model_multi.coef_[0]:.2f} per sq ft")
print(f"Coefficient for Bedrooms (β₂): ${model_multi.coef_[1]:,.2f} per bedroom")
print(f"R² Score: {r2_score(y_multi, y_pred_multi):.3f}")
print(f"\nRegression Equation:")
print(f"Price = {model_multi.intercept_:,.0f} + {model_multi.coef_[0]:.1f} × Size + {model_multi.coef_[1]:,.0f} × Bedrooms")

# Visualizations for multiple regression
plt.subplot(1, 3, 2)
plt.scatter(house_sizes_multi, house_prices_multi, c=bedrooms, cmap='viridis', alpha=0.6)
plt.colorbar(label='Number of Bedrooms')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Multiple Regression Data\n(Color = Bedrooms)')
plt.grid(True, alpha=0.3)

# 3D visualization
ax = plt.subplot(1, 3, 3, projection='3d')
scatter = ax.scatter(house_sizes_multi, bedrooms, house_prices_multi, c=house_prices_multi, cmap='coolwarm', alpha=0.6)
ax.set_xlabel('House Size (sq ft)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price ($)')
ax.set_title('3D View: Multiple Regression')

# Create a mesh for the regression plane
size_range = np.linspace(house_sizes_multi.min(), house_sizes_multi.max(), 10)
bedroom_range = np.linspace(bedrooms.min(), bedrooms.max(), 10)
size_mesh, bedroom_mesh = np.meshgrid(size_range, bedroom_range)
price_mesh = (model_multi.intercept_ + 
              model_multi.coef_[0] * size_mesh + 
              model_multi.coef_[1] * bedroom_mesh)

ax.plot_surface(size_mesh, bedroom_mesh, price_mesh, alpha=0.3, color='red')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

# Example predictions
print("Single Variable Model Predictions:")
test_sizes = [1000, 1500, 2000, 2500]
for size in test_sizes:
    pred_price = model_single.predict([[size]])[0]
    print(f"House size: {size} sq ft → Predicted price: ${pred_price:,.2f}")

print("\nMultiple Variable Model Predictions:")
test_cases = [(1000, 2), (1500, 3), (2000, 3), (2500, 4)]
for size, beds in test_cases:
    pred_price = model_multi.predict([[size, beds]])[0]
    print(f"Size: {size} sq ft, Bedrooms: {beds} → Predicted price: ${pred_price:,.2f}")

print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)
print("1. INTERCEPT interpretation:")
print(f"   - Single model: ${model_single.intercept_:,.0f} (price when size=0)")
print(f"   - Multiple model: ${model_multi.intercept_:,.0f} (price when size=0 AND bedrooms=0)")
print("\n2. The intercept may not be meaningful in real-world context")
print("   (e.g., a house with 0 sq ft doesn't exist)")
print("\n3. Multiple regression captures more variation in the data")
print(f"   - R² improved from {r2_score(y_single, y_pred_single):.3f} to {r2_score(y_multi, y_pred_multi):.3f}")
