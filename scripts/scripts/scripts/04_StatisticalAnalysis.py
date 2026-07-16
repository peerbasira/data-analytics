"""
===========================================================
04_StatisticalAnalysis.py
Statistical Analysis using Python
===========================================================
"""

# Import Libraries
import pandas as pd
import numpy as np

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from scipy.stats import pearsonr

# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("datasets/WineQT.csv")

print("First Five Rows")
print(df.head())

# ---------------------------------
# Select Numerical Columns
# ---------------------------------

numeric_df = df.select_dtypes(include=['number'])

# ---------------------------------
# Mean
# ---------------------------------

print("\nMean")
print(numeric_df.mean())

# ---------------------------------
# Median
# ---------------------------------

print("\nMedian")
print(numeric_df.median())

# ---------------------------------
# Variance
# ---------------------------------

print("\nVariance")
print(numeric_df.var())

# ---------------------------------
# Standard Deviation
# ---------------------------------

print("\nStandard Deviation")
print(numeric_df.std())

# ---------------------------------
# Correlation Analysis
# ---------------------------------

print("\nCorrelation Matrix")
print(numeric_df.corr())

# Pearson Correlation

if len(numeric_df.columns) >= 2:

    r, p = pearsonr(
        numeric_df.iloc[:,0],
        numeric_df.iloc[:,1]
    )

    print("\nPearson Correlation")

    print("Correlation Coefficient:", r)

    print("P-value:", p)

# ---------------------------------
# Independent t-test
# ---------------------------------

if len(numeric_df.columns) >= 2:

    t_stat, p_value = ttest_ind(
        numeric_df.iloc[:,0],
        numeric_df.iloc[:,1]
    )

    print("\nIndependent t-test")

    print("t-statistic:", t_stat)

    print("P-value:", p_value)

# ---------------------------------
# Chi-Square Test
# ---------------------------------

if len(df.columns) >= 2:

    contingency = pd.crosstab(
        df.iloc[:,0],
        df.iloc[:,1]
    )

    chi2, p, dof, expected = chi2_contingency(
        contingency
    )

    print("\nChi-Square Test")

    print("Chi-square:", chi2)

    print("P-value:", p)

# ---------------------------------
# One-Way ANOVA
# ---------------------------------

if len(numeric_df.columns) >= 3:

    F, p = f_oneway(
        numeric_df.iloc[:,0],
        numeric_df.iloc[:,1],
        numeric_df.iloc[:,2]
    )

    print("\nOne-Way ANOVA")

    print("F-value:", F)

    print("P-value:", p)

print("\nStatistical Analysis Completed Successfully")
