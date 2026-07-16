"""
===========================================================
01_EDA.py
Data Loading and Exploratory Data Analysis (EDA)
===========================================================
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

# Change the filename if required
df = pd.read_csv("datasets/train.csv")

# ---------------------------------------------------------
# Display Dataset
# ---------------------------------------------------------

print("\nFirst Five Rows")
print(df.head())

print("\nLast Five Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nDataset Information")
print(df.info())

# ---------------------------------------------------------
# Summary Statistics
# ---------------------------------------------------------

print("\nSummary Statistics")
print(df.describe(include="all"))

# ---------------------------------------------------------
# Missing Values
# ---------------------------------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# ---------------------------------------------------------
# Duplicate Records
# ---------------------------------------------------------

print("\nDuplicate Rows")
print(df.duplicated().sum())

# ---------------------------------------------------------
# Histogram
# ---------------------------------------------------------

df.hist(figsize=(12,10))

plt.suptitle("Histogram of Numerical Features")

plt.show()

print("\nEDA Completed Successfully")
