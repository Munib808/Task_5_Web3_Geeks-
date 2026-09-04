
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("final_model.joblib")

# IMPORTANT:
# This threshold was selected using the development set,
# not the final test set.
FINAL_THRESHOLD = 0.35000000000000003


def engineer_features(X):
    X = X.copy()

    X["age_bucket"] = pd.cut(
        X["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=[
            "<25",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65+"
        ],
        right=False
    ).astype(object)

    X["hours_bin"] = pd.cut(
        X["hours-per-week"],
        bins=[0, 35, 45, 200],
        labels=[
            "part-time(<35)",
            "full-time(35-45)",
            "overtime(>45)"
        ],
        right=False
    ).astype(object)

    X["capital_gain_flag"] = (
        X["capital-gain"] > 0
    ).astype(int)

    X["log_capital_gain"] = np.log1p(
        X["capital-gain"]
    )

    X["capital_loss_flag"] = (
        X["capital-loss"] > 0
    ).astype(int)

    X["log_capital_loss"] = np.log1p(
        X["capital-loss"]
    )

    X["higher_ed"] = (
        X["education-num"] >= 13
    ).astype(int)

    X["edu_hours_interaction"] = (
        X["education-num"] *
        X["hours-per-week"]
    )

    return X


def predict(model, data):
    probabilities = model.predict_proba(data)[:, 1]

    predictions = (
        probabilities >= FINAL_THRESHOLD
    ).astype(int)

    result = data.copy()

    result["probability_>50K"] = probabilities

    result["prediction"] = predictions

    result["predicted_income"] = np.where(
        predictions == 1,
        ">50K",
        "<=50K"
    )

    return result


if __name__ == "__main__":

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # Example raw input.
    example = pd.DataFrame([{
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }])

    result = predict(
        model,
        example
    )

    print(result[
        [
            "probability_>50K",
            "prediction",
            "predicted_income"
        ]
    ])
