"""
===========================================================
03_DataVisualization.py
Data Visualization using Matplotlib and Seaborn
===========================================================
"""

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("datasets/train.csv")

print("First Five Rows")
print(df.head())

# Select numerical columns
numeric_df = df.select_dtypes(include=['number'])

# ---------------------------------
# Histogram
# ---------------------------------

numeric_df.hist(figsize=(12,10))

plt.suptitle("Histogram of Numerical Features")

plt.show()

# ---------------------------------
# Box Plot
# ---------------------------------

plt.figure(figsize=(10,6))

sns.boxplot(data=numeric_df)

plt.title("Box Plot")

plt.xticks(rotation=45)

plt.show()

# ---------------------------------
# Scatter Plot
# ---------------------------------

if len(numeric_df.columns) >= 2:

    plt.figure(figsize=(8,6))

    plt.scatter(
        numeric_df.iloc[:,0],
        numeric_df.iloc[:,1]
    )

    plt.xlabel(numeric_df.columns[0])

    plt.ylabel(numeric_df.columns[1])

    plt.title("Scatter Plot")

    plt.show()

# ---------------------------------
# Correlation Matrix
# ---------------------------------

correlation = numeric_df.corr()

print("\nCorrelation Matrix")

print(correlation)

# ---------------------------------
# Heatmap
# ---------------------------------

plt.figure(figsize=(10,8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

# ---------------------------------
# Pair Plot
# ---------------------------------

sns.pairplot(numeric_df)

plt.show()

print("\nVisualization Completed Successfully")
