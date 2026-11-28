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
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, learning_curve
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


def tune_cutoff_by_source(model, X, y, source, folds=5, cutoffs=np.arange(0.05, 0.96, 0.05)):
    X, y, source = X.reset_index(drop=True), y.reset_index(drop=True), source.reset_index(drop=True)
    strata = y.astype(str) + "_" + source.astype(str)
    oof_scores = np.zeros(len(y))

    for train_idx, val_idx in make_cv(folds).split(X, strata):
        fold_model = fit_with_weights(
            clone(model),
            X.iloc[train_idx],
            y.iloc[train_idx],
            make_sample_weights(source.iloc[train_idx]),
        )
        oof_scores[val_idx] = fold_model.predict_proba(X.iloc[val_idx])[:, 1]

    rows = []
    for cutoff in cutoffs:
        row = {"Cut-off": round(float(cutoff), 2)}
        for name in source.unique():
            mask = (source == name).to_numpy()
            row[f"{name} F1"] = f1_score(y[mask], oof_scores[mask] >= cutoff, zero_division=0)
        rows.append(row)

    table = pd.DataFrame(rows)
    table["Mean F1"] = table.drop(columns="Cut-off").mean(axis=1)
    best_cutoff = table.loc[table["Mean F1"].idxmax(), "Cut-off"]
    return table, best_cutoff


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


def learning_curve_table(model, X, y, folds=5, sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=sizes,
        cv=make_cv(folds),
        scoring="f1",
        n_jobs=-1,
    )
    return pd.DataFrame({
        "Training size": train_sizes,
        "Training F1": train_scores.mean(axis=1),
        "Training Std": train_scores.std(axis=1),
        "Validation F1": val_scores.mean(axis=1),
        "Validation Std": val_scores.std(axis=1),
    })


def describe_fit(table, gap_limit=0.01, score_limit=0.95):
    last = table.iloc[-1]
    gap = last["Training F1"] - last["Validation F1"]
    if last["Validation F1"] < score_limit and gap <= gap_limit:
        return "Underfitting", gap
    if gap > gap_limit:
        return "Overfitting", gap
    return "Good fit", gap


def plot_learning_curve(table, title, ax=None):
    save_dir = RESULTS_DIR / "learning_curves"
    save_dir.mkdir(parents=True, exist_ok=True)

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 4))
    for name, color in [("Training", "tab:blue"), ("Validation", "tab:orange")]:
        mean = table[f"{name} F1"]
        std = table[f"{name} Std"]
        ax.plot(table["Training size"], mean, "o-", color=color, label=f"{name} F1")
        ax.fill_between(table["Training size"], mean - std, mean + std, color=color, alpha=0.15)
    lower = min((table["Training F1"] - table["Training Std"]).min(), (table["Validation F1"] - table["Validation Std"]).min())
    ax.set_ylim(max(0, lower - 0.02), 1.002)
    ax.ticklabel_format(axis="y", useOffset=False)
    fit, gap = describe_fit(table)
    ax.set_title(f"{title}\n{fit} (gap {gap:.4f})")
    ax.set_xlabel("Training size")
    ax.set_ylabel("F1 score")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    if own_figure:
        fig.tight_layout()
        fig.savefig(save_dir / f"{to_file_name(title)}.png", dpi=300)
        plt.show()

