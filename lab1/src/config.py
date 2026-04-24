import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"


def load_config(name):
    config_path = CONFIG_DIR / f"{name}.json"
    with open(config_path, "r") as f:
        return json.load(f)


def project_path(relative_path):
    return BASE_DIR / relative_path
