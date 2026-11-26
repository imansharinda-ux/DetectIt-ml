import inspect
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
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


def make_sample_weights(source):
    counts = source.value_counts()
    return source.map(counts.max() / counts).astype(float).to_numpy()


def supports_sample_weight(model):
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    return "sample_weight" in inspect.signature(estimator.fit).parameters


def fit_with_weights(model, X, y, sample_weight):
    if not supports_sample_weight(model):
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": sample_weight})
    return model.fit(X, y, sample_weight=sample_weight)


def cross_validate_by_source(model, X, y, source, folds=5):
    X, y, source = X.reset_index(drop=True), y.reset_index(drop=True), source.reset_index(drop=True)
    strata = y.astype(str) + "_" + source.astype(str)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(make_cv(folds).split(X, strata), start=1):
        fold_model = fit_with_weights(
            clone(model),
            X.iloc[train_idx],
            y.iloc[train_idx],
            make_sample_weights(source.iloc[train_idx]),
        )
        for name in source.unique():
            val_idx_source = val_idx[source.iloc[val_idx].to_numpy() == name]
            scores = evaluate_model(fold_model, X.iloc[val_idx_source], y.iloc[val_idx_source])
            rows.append({"Fold": fold, "Source": name, **scores})

    fold_scores = pd.DataFrame(rows)
    table = pd.DataFrame(index=list(METRICS.values()))
    for name in source.unique():
        scores = fold_scores[fold_scores["Source"] == name][list(METRICS.values())]
        table[f"{name} Mean"] = scores.mean()
        table[f"{name} Std"] = scores.std()
    return table


def tune_by_source(model, param_grid, X, y, source, folds=3):
    candidates = list(ParameterGrid(param_grid))
    rows = []
    for params in candidates:
        scores = cross_validate_by_source(clone(model).set_params(**params), X, y, source, folds)
        rows.append({name: scores.loc["F1", f"{name} Mean"] for name in source.unique()})

    table = pd.DataFrame(rows)
    table.columns = [f"{name} F1" for name in table.columns]
    table["Mean F1"] = table.mean(axis=1)
    settings = pd.DataFrame(candidates).astype(object).where(pd.DataFrame(candidates).notna(), "None")
    table = pd.concat([settings, table], axis=1).sort_values("Mean F1", ascending=False)
    best_params = candidates[table.index[0]]
    return table.reset_index(drop=True), best_params


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
