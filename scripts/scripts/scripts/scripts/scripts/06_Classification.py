"""
===========================================================
06_Classification.py
Classification using Machine Learning
===========================================================
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("datasets/heart.csv")

print(df.head())

# ---------------------------------
# Remove Missing Values
# ---------------------------------

df = df.dropna()

# ---------------------------------
# Convert Categorical Columns
# ---------------------------------

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ---------------------------------
# Features and Target
# ---------------------------------

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

# ---------------------------------
# Train-Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# Models
# ---------------------------------

models = {

    "Logistic Regression":
    LogisticRegression(max_iter=1000),

    "KNN":
    KNeighborsClassifier(),

    "Decision Tree":
    DecisionTreeClassifier(),

    "Support Vector Machine":
    SVC()

}

# ---------------------------------
# Train & Evaluate
# ---------------------------------

for name, model in models.items():

    print("\n==============================")

    print(name)

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            prediction
        )
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            prediction,
            average="weighted"
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            prediction,
            average="weighted"
        )
    )

    print(
        "F1 Score:",
        f1_score(
            y_test,
            prediction,
            average="weighted"
        )
    )

    print(
        "Confusion Matrix"
    )

    print(
        confusion_matrix(
            y_test,
            prediction
        )
    )

print("\nClassification Completed Successfully")
