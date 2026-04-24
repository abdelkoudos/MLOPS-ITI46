import joblib
import pandas as pd

from config import load_config, project_path
from data import FEATURE_COLUMNS, ID_COLUMN, RAW_DIR

DATA_CONFIG = load_config("data")
PRED_CONFIG = load_config("pred")
PIPELINE_CONFIG = load_config("pipeline")
ARTIFACTS_DIR = project_path(PIPELINE_CONFIG["artifacts_dir"])
PREDICTIONS_PATH = ARTIFACTS_DIR / PRED_CONFIG["predictions_file"]


def load_model():
    model_path = ARTIFACTS_DIR / PIPELINE_CONFIG["model_file"]
    return joblib.load(model_path)


def load_test_data():
    test_path = RAW_DIR / DATA_CONFIG["test_file"]
    return pd.read_csv(test_path)


def predict(model, test_df):
    X_test = test_df[FEATURE_COLUMNS]
    return model.predict(X_test)


def save_predictions(test_df, predictions, model_name=None):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    output = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN],
            "Survived": predictions,
        }
    )
    if model_name is not None:
        predictions_filename = PRED_CONFIG[
            "per_model_predictions_template"
        ].format(model_name=model_name)
        output.to_csv(ARTIFACTS_DIR / predictions_filename, index=False)

    output.to_csv(PREDICTIONS_PATH, index=False)


def main():
    model = load_model()
    test_df = load_test_data()
    predictions = predict(model, test_df)
    save_predictions(test_df, predictions)
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
