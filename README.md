# 📊 Data Analytics Using Python

## Project Overview

This repository contains implementations of fundamental **Data Analytics** and **Machine Learning** techniques using Python. It demonstrates the complete data analytics workflow, including data loading, preprocessing, visualization, statistical analysis, predictive modeling, clustering, dimensionality reduction, time series forecasting, and an end-to-end analytics pipeline.

The project was developed as part of an M.Tech Data Analytics using Python coursework.

---

# Project Structure

```
Data-Analytics-Using-Python/
│
├── datasets/
│   ├── AirPassengers.csv
│   ├── heart.csv
│   ├── housing.csv
│   ├── Iris.csv
│   ├── Mall_Customers.csv
│   ├── train.csv
│   ├── WineQT.csv
│   └── healthcare-dataset-stroke-data.csv (optional)
│
├── scripts/
│   ├── 01_EDA.py
│   ├── 02_DataPreprocessing.py
│   ├── 03_DataVisualization.py
│   ├── 04_StatisticalAnalysis.py
│   ├── 05_Regression.py
│   ├── 06_Classification.py
│   ├── 07_Clustering.py
│   ├── 08_PCA.py
│   ├── 09_TimeSeriesAnalysis.py
│   └── 10_MiniProject.py
│
├── README.md
└── requirements.txt
```

---

# Project Modules

### 1. Data Loading and Exploration (EDA)

* Load CSV datasets using Pandas
* Display dataset information
* Summary statistics
* Data types inspection
* Missing value detection
* Duplicate detection

**Dataset:** Titanic (`train.csv`)

---

### 2. Data Preprocessing and Cleaning

* Handle missing values
* Remove duplicate records
* Label Encoding
* Feature Scaling
* Normalization
* Standardization
* Outlier Detection

**Dataset:** Titanic (`train.csv`)

---

### 3. Data Visualization

* Histogram
* Scatter Plot
* Box Plot
* Heatmap
* Correlation Matrix
* Pair Plot

**Dataset:** Iris (`Iris.csv`)

---

### 4. Statistical Analysis

* Mean
* Median
* Variance
* Standard Deviation
* Pearson Correlation
* Independent t-Test
* Chi-Square Test
* One-Way ANOVA

**Dataset:** Wine Quality (`WineQT.csv`)

---

### 5. Regression Analysis

* Linear Regression
* Multiple Linear Regression
* Model Evaluation

  * R² Score
  * MSE
  * RMSE
  * MAE

**Dataset:** California Housing (`housing.csv`)

---

### 6. Classification Techniques

* Logistic Regression
* k-Nearest Neighbors (kNN)
* Decision Tree
* Support Vector Machine (SVM)

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

**Dataset:** Heart Disease (`heart.csv`)

---

### 7. Clustering Algorithms

* K-Means Clustering
* Hierarchical Clustering
* Silhouette Score
* Dendrogram Analysis

**Dataset:** Mall Customers (`Mall_Customers.csv`)

---

### 8. Dimensionality Reduction

* Principal Component Analysis (PCA)
* 2D Visualization
* 3D Visualization

**Dataset:** Iris (`Iris.csv`)

---

### 9. Time Series Analysis

* Time Series Decomposition
* Moving Average
* ARIMA Forecasting

**Dataset:** AirPassengers (`AirPassengers.csv`)

---

### 10. Mini Project

End-to-End Data Analytics Pipeline

Includes:

* Data Loading
* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Machine Learning Model
* Prediction
* Model Evaluation
* Result Interpretation

**Dataset:** Heart Disease (`heart.csv`)

---

# Python Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* scipy
* statsmodels
* openpyxl

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Data-Analytics-Using-Python.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run any script:

```bash
python scripts/01_EDA.py
```

---

# Datasets Used

| Module               | Dataset                               |
| -------------------- | ------------------------------------- |
| EDA                  | Titanic (`train.csv`)                 |
| Data Preprocessing   | Titanic (`train.csv`)                 |
| Data Visualization   | Iris (`Iris.csv`)                     |
| Statistical Analysis | Wine Quality (`WineQT.csv`)           |
| Regression           | California Housing (`housing.csv`)    |
| Classification       | Heart Disease (`heart.csv`)           |
| Clustering           | Mall Customers (`Mall_Customers.csv`) |
| PCA                  | Iris (`Iris.csv`)                     |
| Time Series          | AirPassengers (`AirPassengers.csv`)   |
| Mini Project         | Heart Disease (`heart.csv`)           |

---

# Dataset Sources

* Titanic Dataset: https://www.kaggle.com/competitions/titanic
* Iris Dataset: https://www.kaggle.com/datasets/uciml/iris
* Wine Quality Dataset: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
* California Housing Dataset: https://www.kaggle.com/datasets/camnugent/california-housing-prices
* Heart Disease Dataset: https://www.kaggle.com/datasets/amineipad/heart-disease-dataset
* Mall Customer Segmentation Dataset: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
* Airline Passenger Dataset: https://www.kaggle.com/datasets/vedatgul/airline-passengers

---

# Learning Outcomes

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Statistical Analysis
* Data Visualization
* Regression
* Classification
* Clustering
* Principal Component Analysis
* Time Series Forecasting
* End-to-End Machine Learning Pipeline

---

# Author

**Lubna Shafi**

M.Tech Computer Science and Engineering

Course: Data Analytics Using Python
