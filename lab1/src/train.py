import joblib
from config import load_config, project_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import prepare_train_test_data

DATA_CONFIG = load_config("data")
TRAIN_CONFIG = load_config("train")
PIPELINE_CONFIG = load_config("pipeline")
ARTIFACTS_DIR = project_path(PIPELINE_CONFIG["artifacts_dir"])
AVAILABLE_MODELS = tuple(TRAIN_CONFIG["models"].keys())
DEFAULT_MODEL = TRAIN_CONFIG["default_model"]
NUMERIC_FEATURES = DATA_CONFIG["numeric_features"]
CATEGORICAL_FEATURES = DATA_CONFIG["categorical_features"]


def build_one_hot_encoder():
    handle_unknown = TRAIN_CONFIG["preprocessing"]["one_hot_handle_unknown"]
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)


def build_preprocessor():
    preprocessing_config = TRAIN_CONFIG["preprocessing"]
    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy=preprocessing_config["numeric_imputer_strategy"]
            ),
        ),
    ]
    if preprocessing_config["scale_numeric"]:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=preprocessing_config[
                        "categorical_imputer_strategy"
                    ]
                ),
            ),
            ("encoder", build_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model(model_name):
    model_config = TRAIN_CONFIG["models"].get(model_name)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = model_config["class"]
    model_params = model_config["params"]

    if model_class == "LogisticRegression":
        estimator = LogisticRegression(**model_params)

    elif model_class == "RandomForestClassifier":
        estimator = RandomForestClassifier(**model_params)

    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def train_model(df, model_name=DEFAULT_MODEL):
    X_train, X_val, y_train, y_val = prepare_train_test_data(df)

    model = build_model(model_name)
    model.fit(X_train, y_train)

    return model, X_val, y_val


def save_model(model, model_name=DEFAULT_MODEL):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = PIPELINE_CONFIG["per_model_model_template"].format(
        model_name=model_name
    )
    joblib.dump(model, ARTIFACTS_DIR / model_filename)
    joblib.dump(model, ARTIFACTS_DIR / PIPELINE_CONFIG["model_file"])
