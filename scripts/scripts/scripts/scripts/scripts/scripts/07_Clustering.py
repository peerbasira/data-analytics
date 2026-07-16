"""
===========================================================
07_Clustering.py
K-Means and Hierarchical Clustering
===========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------------
# Load Dataset
# -------------------------------------

df = pd.read_csv("datasets/Mall_Customers.csv")

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
# K-Means Clustering
# -------------------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X)

print("\nK-Means Silhouette Score")

print(silhouette_score(X, labels))

# -------------------------------------
# Scatter Plot
# -------------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    X[:,0],
    X[:,1],
    c=labels
)

plt.title("K-Means Clustering")

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.show()

# -------------------------------------
# Hierarchical Clustering
# -------------------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=3
)

hierarchical_labels = hierarchical.fit_predict(X)

print("\nHierarchical Silhouette Score")

print(
    silhouette_score(
        X,
        hierarchical_labels
    )
)

# -------------------------------------
# Dendrogram
# -------------------------------------

plt.figure(figsize=(10,6))

linkage_matrix = linkage(
    X,
    method="ward"
)

dendrogram(linkage_matrix)

plt.title("Hierarchical Clustering Dendrogram")

plt.xlabel("Samples")

plt.ylabel("Distance")

plt.show()

print("\nClustering Completed Successfully")
