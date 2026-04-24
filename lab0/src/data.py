from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data"
TARGET_COLUMN = "Survived"
FEATURE_COLUMNS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]
MODEL_COLUMNS = [TARGET_COLUMN] + FEATURE_COLUMNS


def load_data():
    data_path = RAW_DIR / "train.csv"
    return pd.read_csv(data_path)


def split_train_test(X, y=None, test_size=0.2, random_state=42):
    if y is None:
        return train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
            stratify=X[TARGET_COLUMN],
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def prepare_train_test_data(df, test_size=0.2, random_state=42):
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
