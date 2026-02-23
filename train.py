import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Generate synthetic dataset
np.random.seed(42)

data_size = 500

square_feet = np.random.randint(500, 3000, data_size)
bedrooms = np.random.randint(1, 5, data_size)
bathrooms = np.random.randint(1, 4, data_size)

price = (
    square_feet * 150 +
    bedrooms * 10000 +
    bathrooms * 8000 +
    np.random.randint(-20000, 20000, data_size)
)

data = pd.DataFrame({
    "square_feet": square_feet,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "price": price
})

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
pickle.dump(model, open("house_price_model.pkl", "wb"))

print("Model saved successfully.")
