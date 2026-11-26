import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import build_all_url_features, build_clean_url_features

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


def load_url_part(url_file, label_file):
    urls = pd.read_csv(PROCESSED_DIR / url_file).iloc[:, 0]
    y = pd.read_csv(PROCESSED_DIR / label_file, usecols=[TARGET])[TARGET]
    return urls, y


def load_url_experiment(clean_urls=False):
    build_features = build_clean_url_features if clean_urls else build_all_url_features
    phi_train_urls, phi_train_y = load_url_part("train_urls.csv", "train.csv")
    phi_test_urls, phi_test_y = load_url_part("test_urls.csv", "test.csv")
    kaggle_train_urls, kaggle_train_y = load_url_part("external_train_urls.csv", "external_train.csv")
    kaggle_test_urls, kaggle_test_y = load_url_part("external_test_urls.csv", "external_test.csv")

    train_urls = pd.concat([phi_train_urls, kaggle_train_urls], ignore_index=True)
    y_train = pd.concat([phi_train_y, kaggle_train_y], ignore_index=True)
    source = pd.Series(["PhiUSIIL"] * len(phi_train_urls) + ["Kaggle"] * len(kaggle_train_urls), name="Source")
    X_train = build_features(train_urls)

    test_sets = {
        "PhiUSIIL test set": (build_features(phi_test_urls), phi_test_y),
        "Kaggle test set": (build_features(kaggle_test_urls), kaggle_test_y),
    }
    return X_train, y_train, source, test_sets


def sample_rows(X, y, size, random_state=42, groups=None):
    if size is None or size >= len(X):
        return (X, y) if groups is None else (X, y, groups)
    stratify = y if groups is None else y.astype(str) + "_" + groups.astype(str)
    index, _ = train_test_split(X.index, train_size=size, stratify=stratify, random_state=random_state)
    if groups is None:
        return X.loc[index], y.loc[index]
    return X.loc[index], y.loc[index], groups.loc[index]
