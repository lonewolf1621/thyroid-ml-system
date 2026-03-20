import pandas as pd
import joblib

from pipelines.preprocessing import preprocess_data
from src.model_registry import get_model


def train_tabular_model(data_path, target_column, model_name):

    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df,
        target_column
    )

    model = get_model(model_name)

    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Training complete")