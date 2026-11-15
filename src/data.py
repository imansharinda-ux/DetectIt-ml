import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import build_url_features

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TARGET = "label"


def load_feature_groups():
    with open(PROCESSED_DIR / "feature_names.json") as f:
        return json.load(f)


def get_feature_sets():
    groups = load_feature_groups()
    return {
        "All features": groups["all"],
        "Original features only": groups["original"],
        "Without URLSimilarityIndex": [col for col in groups["all"] if col != "URLSimilarityIndex"],
    }


def load_train_test():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train.drop(columns=TARGET), test.drop(columns=TARGET), train[TARGET], test[TARGET]


def load_url_experiment():
    train_urls = pd.read_csv(PROCESSED_DIR / "train_urls.csv")["URL"]
    test_urls = pd.read_csv(PROCESSED_DIR / "test_urls.csv")["URL"]
    y_train = pd.read_csv(PROCESSED_DIR / "train.csv", usecols=[TARGET])[TARGET]
    y_test = pd.read_csv(PROCESSED_DIR / "test.csv", usecols=[TARGET])[TARGET]
    external = pd.read_csv(PROCESSED_DIR / "external_test.csv")

    X_train = build_url_features(train_urls)
    X_test = build_url_features(test_urls)
    X_external = external[X_train.columns]
    return X_train, X_test, X_external, y_train, y_test, external[TARGET]


def sample_rows(X, y, size, random_state=42):
    if size is None or size >= len(X):
        return X, y
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=random_state)
    return X_sample, y_sample
