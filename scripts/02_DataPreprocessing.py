"""
===========================================================
02_DataPreprocessing.py
Data Preprocessing and Cleaning
===========================================================
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("datasets/train.csv")

print("Original Dataset")
print(df.head())

# ---------------------------------
# Handle Missing Values
# ---------------------------------

numeric_columns = df.select_dtypes(include=['number']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values")
print(df.isnull().sum())

# ---------------------------------
# Remove Duplicates
# ---------------------------------

df = df.drop_duplicates()

print("\nDuplicates Removed")

# ---------------------------------
# Normalization
# ---------------------------------

minmax = MinMaxScaler()

df[numeric_columns] = minmax.fit_transform(
    df[numeric_columns]
)

print("\nNormalized Data")
print(df.head())

# ---------------------------------
# Standardization
# ---------------------------------

standard = StandardScaler()

scaled = standard.fit_transform(
    df[numeric_columns]
)

print("\nStandardization Completed")

# ---------------------------------
# Label Encoding
# ---------------------------------

encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nEncoded Dataset")
print(df.head())

# ---------------------------------
# Outlier Detection (IQR)
# ---------------------------------

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(f"{col}: {len(outliers)} outliers")

print("\nPreprocessing Completed Successfully")
