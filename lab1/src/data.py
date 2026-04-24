import pandas as pd
from sklearn.model_selection import train_test_split

from config import load_config, project_path

DATA_CONFIG = load_config("data")
RAW_DIR = project_path(DATA_CONFIG["raw_dir"])
TARGET_COLUMN = DATA_CONFIG["target_column"]
ID_COLUMN = DATA_CONFIG["id_column"]
FEATURE_COLUMNS = DATA_CONFIG["feature_columns"]
MODEL_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS


def load_data():
    data_path = RAW_DIR / DATA_CONFIG["train_file"]
    return pd.read_csv(data_path)


def split_train_test(X, y=None, test_size=0.2, random_state=42):
    stratify_enabled = DATA_CONFIG["split"]["stratify"]

    if y is None:
        stratify = X[TARGET_COLUMN] if stratify_enabled else None
        return train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    stratify = y if stratify_enabled else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def prepare_train_test_data(df, test_size=None, random_state=None):
    split_config = DATA_CONFIG["split"]
    test_size = split_config["test_size"] if test_size is None else test_size
    random_state = (
        split_config["random_state"] if random_state is None else random_state
    )

    train_df, test_df = split_train_test(
        df[MODEL_COLUMNS],
        test_size=test_size,
        random_state=random_state,
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test
