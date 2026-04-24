import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def load_model():
    model_path = ARTIFACTS_DIR / "model.pkl"
    return joblib.load(model_path)


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "auc": roc_auc_score(y_val, y_proba),
    }

    return metrics


def save_metrics(metrics, model_name=None):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if model_name is not None:
        with open(ARTIFACTS_DIR / f"{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
