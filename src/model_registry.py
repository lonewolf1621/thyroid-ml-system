from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_model(model_name):

    models = {
        "random_forest": RandomForestClassifier(),
        "logistic_regression": LogisticRegression()
    }

    if model_name not in models:
        raise ValueError("Model not supported")

    return models[model_name]