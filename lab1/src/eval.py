import json

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score

from config import load_config, project_path

EVAL_CONFIG = load_config("eval")
PIPELINE_CONFIG = load_config("pipeline")
ARTIFACTS_DIR = project_path(PIPELINE_CONFIG["artifacts_dir"])


def load_model():
    model_path = ARTIFACTS_DIR / PIPELINE_CONFIG["model_file"]
    return joblib.load(model_path)


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {}
    if "accuracy" in EVAL_CONFIG["metrics"]:
        metrics["accuracy"] = accuracy_score(y_val, y_pred)
    if "auc" in EVAL_CONFIG["metrics"]:
        metrics["auc"] = roc_auc_score(y_val, y_proba)

    return metrics


def save_metrics(metrics, model_name=None):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if model_name is not None:
        metrics_filename = EVAL_CONFIG["per_model_metrics_template"].format(
            model_name=model_name
        )
        with open(ARTIFACTS_DIR / metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)

    with open(ARTIFACTS_DIR / EVAL_CONFIG["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=4)
