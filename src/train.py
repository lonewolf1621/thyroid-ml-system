import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


DATA_PATH = "data/hypothyroid.csv"
TARGET_COLUMN = "binaryClass"

MODEL_VERSION = "v1"

mlflow.set_experiment("thyroid-disease-classification")


def train():

    df = pd.read_csv(DATA_PATH)

    df.replace("?", np.nan, inplace=True)

    df.drop_duplicates(inplace=True)

    df = df.dropna(axis=1, how="all")

    y = df[TARGET_COLUMN]

    X = df.drop(columns=[TARGET_COLUMN])

    y = y.apply(lambda x: 1 if x == "P" else 0)

    for col in X.columns:

        try:
            X[col] = pd.to_numeric(X[col])
        except:
            pass

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    os.makedirs(f"models/{MODEL_VERSION}", exist_ok=True)

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            clf = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            clf.fit(X_train, y_train)

            y_prob = clf.predict_proba(X_test)[:, 1]

            roc = roc_auc_score(y_test, y_prob)

            mlflow.log_param("model_name", name)

            mlflow.log_metric("roc_auc", roc)

            mlflow.sklearn.log_model(clf, name)

            joblib.dump(clf, f"models/{MODEL_VERSION}/{name}.pkl")

            print(f"{name} ROC-AUC:", roc)


if __name__ == "__main__":
    train()