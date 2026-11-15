import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
RANDOM_STATE = 42

METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
}


def make_scaled_pipeline(model):
    return make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
        model,
    )


def make_cv(folds=5):
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_model(model, X, y, folds=5):
    scores = cross_validate(model, X, y, cv=make_cv(folds), scoring=list(METRICS), n_jobs=-1)
    table = pd.DataFrame({name: scores[f"test_{key}"] for key, name in METRICS.items()})
    return table.agg(["mean", "std"]).T.rename(columns={"mean": "Mean", "std": "Std"})


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_predictions(y_true, y_pred, y_score=None):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
    }


def evaluate_model(model, X, y):
    return evaluate_predictions(y, model.predict(X), get_scores(model, X))


def to_file_name(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def plot_confusion_matrix(y_true, y_pred, title):
    save_dir = RESULTS_DIR / "confusion_matrices"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Legitimate", "Phishing"],
        cmap="Blues",
        values_format=",",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_dir / f"{to_file_name(title)}.png", dpi=300)
    plt.show()


def save_results(model_name, results):
    save_dir = RESULTS_DIR / "metrics"
    save_dir.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(results)
    table.insert(0, "Model", model_name)
    table.to_csv(save_dir / f"{to_file_name(model_name)}.csv", index=False)
    return table
