from pathlib import Path

import joblib
import pandas as pd

from data import FEATURE_COLUMNS, RAW_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"


def load_model():
    model_path = ARTIFACTS_DIR / "model.pkl"
    return joblib.load(model_path)


def load_test_data():
    test_path = RAW_DIR / "test.csv"
    return pd.read_csv(test_path)


def predict(model, test_df):
    X_test = test_df[FEATURE_COLUMNS]
    return model.predict(X_test)


def save_predictions(test_df, predictions, model_name=None):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    output = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"],
            "Survived": predictions,
        }
    )
    if model_name is not None:
        output.to_csv(ARTIFACTS_DIR / f"{model_name}_predictions.csv", index=False)

    output.to_csv(PREDICTIONS_PATH, index=False)


def main():
    model = load_model()
    test_df = load_test_data()
    predictions = predict(model, test_df)
    save_predictions(test_df, predictions)
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
