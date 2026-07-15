
# Healthcare Dataset

# Import libraries

import pandas as pd
import numpy as np
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

# 1. Data Collection / Loading


df = pd.read_csv("heart.csv")

print("First 5 Rows")
print(df.head())

# 2. Data Preprocessing


print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Features and target

X = df.drop("target", axis=1)

y = df["target"]

# Scale features

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Exploratory Data Analysis


print("\nSummary Statistics")
print(df.describe())

print("\nData Types")
print(df.dtypes)

# 4. Visualization


# Histogram

df.hist(figsize=(12,10))

plt.show()

# Correlation heatmap

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True
)

plt.title("Correlation Heatmap")

plt.show()

# Target distribution

sns.countplot(
    x='target',
    data=df
)

plt.title(
    "Heart Disease Distribution"
)

plt.show()

# 5. Model Building


model = LogisticRegression()

model.fit(
    X_train,
    y_train
)


# 6. Model Evaluation


y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nAccuracy:",
    accuracy
)

print(
    "\nConfusion Matrix"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print(
    "\nClassification Report"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# 7. Interpretation


print(
"\nInterpretation:"
)

print(
"The model predicts whether a patient has heart disease."
)

print(
"Higher accuracy indicates better prediction performance."
)