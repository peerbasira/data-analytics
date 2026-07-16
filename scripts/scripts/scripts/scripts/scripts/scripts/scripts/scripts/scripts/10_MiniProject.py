"""
===========================================================
10_MiniProject.py
End-to-End Data Analytics Pipeline
Heart Disease Prediction
===========================================================
"""

# ----------------------------------------------------
# Import Libraries
# ----------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

df = pd.read_csv("datasets/heart.csv")

print("\n========== FIRST FIVE RECORDS ==========\n")
print(df.head())

# ----------------------------------------------------
# Dataset Information
# ----------------------------------------------------

print("\n========== DATASET INFO ==========\n")
print(df.info())

print("\n========== SUMMARY ==========\n")
print(df.describe())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# ----------------------------------------------------
# Remove Missing Values
# ----------------------------------------------------

df = df.dropna()

# ----------------------------------------------------
# Remove Duplicate Records
# ----------------------------------------------------

df = df.drop_duplicates()

# ----------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# ----------------------------------------------------
# Histogram
# ----------------------------------------------------

df.hist(figsize=(12,10))

plt.show()

# ----------------------------------------------------
# Target Distribution
# ----------------------------------------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x="target",
    data=df
)

plt.title("Heart Disease Distribution")

plt.show()

# ----------------------------------------------------
# Feature Selection
# ----------------------------------------------------

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

# ----------------------------------------------------
# Feature Scaling
# ----------------------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ----------------------------------------------------
# Train-Test Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)

# ----------------------------------------------------
# Logistic Regression Model
# ----------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(
    X_train,
    y_train
)

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

prediction = model.predict(X_test)

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

print("\nAccuracy")

print(
    accuracy_score(
        y_test,
        prediction
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        prediction
    )
)

# ----------------------------------------------------
# Interpretation
# ----------------------------------------------------

print("\n===================================")
print("PROJECT INTERPRETATION")
print("===================================")

print("""
The machine learning model predicts
whether a patient is likely to have
heart disease based on medical
attributes.

The workflow includes

✔ Data Loading

✔ Data Cleaning

✔ Data Visualization

✔ Exploratory Data Analysis

✔ Feature Scaling

✔ Model Building

✔ Prediction

✔ Evaluation

This demonstrates a complete
end-to-end Data Analytics Pipeline.
""")
