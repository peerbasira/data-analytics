"""
===========================================================
05_Regression.py
Linear and Multiple Linear Regression
===========================================================
"""

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("datasets/housing.csv")

print("First Five Rows")
print(df.head())

# ---------------------------------
# Data Cleaning
# ---------------------------------

# Keep only numerical columns
df = df.select_dtypes(include=["number"])

# Remove missing values
df = df.dropna()

# ---------------------------------
# Features and Target
# ---------------------------------

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

# ---------------------------------
# Train-Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# Model
# ---------------------------------

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# ---------------------------------
# Prediction
# ---------------------------------

y_pred = model.predict(X_test)

# ---------------------------------
# Evaluation
# ---------------------------------

print("\nR2 Score")
print(r2_score(y_test, y_pred))

mse = mean_squared_error(
    y_test,
    y_pred
)

print("\nMean Squared Error")
print(mse)

rmse = np.sqrt(mse)

print("\nRoot Mean Squared Error")
print(rmse)

print("\nMean Absolute Error")
print(mean_absolute_error(
    y_test,
    y_pred
))

print("\nRegression Completed Successfully")
