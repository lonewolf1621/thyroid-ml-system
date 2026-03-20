import os
import joblib

MODEL_VERSION = "v1"

MODEL_DIR = f"models/{MODEL_VERSION}"


def load_models():

    models = {}

    for file in os.listdir(MODEL_DIR):

        if file.endswith(".pkl"):

            name = file.replace(".pkl", "")

            models[name] = joblib.load(
                os.path.join(MODEL_DIR, file)
            )

    return models