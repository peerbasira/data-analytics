"""
===========================================================
08_PCA.py
Principal Component Analysis (PCA)
===========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

# -------------------------------------
# Load Dataset
# -------------------------------------

df = pd.read_csv("datasets/Iris.csv")

print(df.head())

# -------------------------------------
# Keep Numeric Columns
# -------------------------------------

df = df.select_dtypes(include=["number"])

df = df.dropna()

# -------------------------------------
# Standardization
# -------------------------------------

scaler = StandardScaler()

X = scaler.fit_transform(df)

# -------------------------------------
# PCA - 2 Components
# -------------------------------------

pca = PCA(
    n_components=2
)

principal = pca.fit_transform(X)

pca_df = pd.DataFrame(
    principal,
    columns=["PC1","PC2"]
)

print("\nExplained Variance Ratio")

print(
    pca.explained_variance_ratio_
)

# -------------------------------------
# 2D Plot
# -------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"]
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("PCA 2D Visualization")

plt.show()

# -------------------------------------
# PCA 3D
# -------------------------------------

pca3 = PCA(
    n_components=3
)

principal3 = pca3.fit_transform(X)

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.scatter(
    principal3[:,0],
    principal3[:,1],
    principal3[:,2]
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.title("PCA 3D Visualization")

plt.show()

print("\nPCA Completed Successfully")
