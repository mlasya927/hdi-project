"""
Epic 2 & 4: Import Libraries + Data Preprocessing / Label Encoding
--------------------------------------------------------------------
Loads the raw HDI dataset, performs cleaning, encodes the target
label (HDI_Category) using LabelEncoder, and scales the numeric
features using StandardScaler.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(csv_path="data/hdi_dataset.csv"):
    # ---- Load dataset ----
    df = pd.read_csv(csv_path)

    # ---- Basic cleaning ----
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # ---- Features & Target ----
    feature_cols = [
        "Life_Expectancy",
        "Mean_Years_Schooling",
        "Expected_Years_Schooling",
        "GNI_per_Capita",
    ]
    X = df[feature_cols].copy()
    y_raw = df["HDI_Category"].copy()

    # ---- Label Encoding for target (Very High/High/Medium/Low -> 0,1,2,3) ----
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # ---- Feature Scaling ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    return X_scaled, y, scaler, label_encoder, feature_cols


if __name__ == "__main__":
    X, y, scaler, le, cols = load_and_preprocess()
    print("Preprocessing complete.")
    print("Feature columns:", cols)
    print("Encoded classes:", dict(zip(le.classes_, le.transform(le.classes_))))
    print(X.head())
