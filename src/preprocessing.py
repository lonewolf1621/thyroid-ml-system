import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str):
    df = pd.read_csv(path)

    # Replace ? with NaN
    df.replace("?", np.nan, inplace=True)

    return df


def preprocess_data(df, target_column):

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Separate target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Convert boolean-like columns (t/f) to 1/0
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].replace({"t": 1, "f": 0})

    # Convert numeric columns properly
    X = X.apply(pd.to_numeric, errors='ignore')

    # Fill numeric missing values with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # Fill remaining categorical NaNs with mode
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna(X[col].mode()[0])
            X[col] = X[col].astype("category").cat.codes

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)