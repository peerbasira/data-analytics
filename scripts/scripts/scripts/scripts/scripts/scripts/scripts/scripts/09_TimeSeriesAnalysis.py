"""
===========================================================
09_TimeSeriesAnalysis.py
Time Series Analysis using ARIMA
===========================================================
"""

# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("datasets/AirPassengers.csv")

print("First Five Rows")
print(df.head())

# ---------------------------------
# Convert Date Column
# ---------------------------------

df["month"] = pd.to_datetime(df["month"])

df.set_index(
    "month",
    inplace=True
)

# Target Variable

series = df["total_passengers"]

# ---------------------------------
# Original Time Series
# ---------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    series,
    color="blue"
)

plt.title("Original Time Series")

plt.xlabel("Year")

plt.ylabel("Passengers")

plt.grid(True)

plt.show()

# ---------------------------------
# Moving Average
# ---------------------------------

moving_average = series.rolling(
    window=12
).mean()

plt.figure(figsize=(10,6))

plt.plot(
    series,
    label="Original"
)

plt.plot(
    moving_average,
    label="Moving Average"
)

plt.legend()

plt.title("Moving Average")

plt.show()

# ---------------------------------
# Time Series Decomposition
# ---------------------------------

decomposition = seasonal_decompose(
    series,
    model="additive",
    period=12
)

decomposition.plot()

plt.show()

# ---------------------------------
# ARIMA Model
# ---------------------------------

model = ARIMA(
    series,
    order=(1,1,1)
)

model_fit = model.fit()

forecast = model_fit.forecast(
    steps=12
)

print("\nForecast Values")

print(forecast)

# ---------------------------------
# Forecast Plot
# ---------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    series,
    label="Original"
)

plt.plot(
    forecast,
    label="Forecast",
    color="red"
)

plt.legend()

plt.title("ARIMA Forecast")

plt.show()

print("\nTime Series Analysis Completed Successfully")
