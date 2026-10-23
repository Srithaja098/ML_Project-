"""Train KNN, Decision Tree and RBF-SVM on AI4I 2020.

Hyperparameters are chosen by stratified 5-fold CV on the 80% training split only; the 20% test
split is never touched here (it is reserved for the shared evaluation phase).

Usage (from the repo root):
    python -m classification.train                       # all three models
    python -m classification.train --models knn svm_rbf  # a subset
"""
import argparse
import importlib.metadata
import json
import platform

import joblib
import matplotlib
matplotlib.use("Agg")  # write PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             make_scorer, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classification.data import (CATEGORICAL, ID_COLUMNS, LEAKAGE_COLUMNS, NUMERIC, TARGET,
                                         describe_dataset, load_dataset, make_preprocessor,
                                         split_features_target, stratified_split)
from config import ARTIFACTS_DIR, OUTPUTS_DIR, SEED, TEST_SIZE

CV_FOLDS = 5

# Failures are ~3% of rows, so accuracy is misleading (always predicting "no failure" scores ~97%).
# Hyperparameters are picked by F1 of the failure class (balances precision and recall at the default
# decision threshold). Accuracy, precision, recall, PR-AUC, ROC-AUC and balanced accuracy are recorded
# for every candidate too. Precision/recall/F1 are computed for the failure class (label 1); a fold where
# a candidate predicts no failures scores 0 instead of raising a warning.
FAILURE_LABEL = 1
SCORING = {
    "f1": make_scorer(f1_score, pos_label=FAILURE_LABEL, zero_division=0),
    "precision": make_scorer(precision_score, pos_label=FAILURE_LABEL, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=FAILURE_LABEL, zero_division=0),
    "accuracy": "accuracy",
    "average_precision": "average_precision",  # binary: positive class is classes_[1] = 1 (failure)
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
}
REFIT_METRIC = "f1"

# ROC curves for the SVM work without probabilities (via decision_function). probability=True adds an
# internal 5-fold Platt-scaling fit, roughly 5x the training time, so only enable it if
# predict_proba is really needed later.
SVM_PROBABILITY = False

# grid: candidates searched by CV. plot: (x-axis param, one line per value of this param) for the curve.
MODELS = {
    "knn": dict(
        estimator=KNeighborsClassifier(),
        scale=True,
        grid={"clf__n_neighbors": list(range(1, 32, 2)), "clf__weights": ["uniform", "distance"]},
        plot=("clf__n_neighbors", "clf__weights"),
    ),
    "decision_tree": dict(
        estimator=DecisionTreeClassifier(random_state=SEED),
        scale=False,
        grid={"clf__max_depth": [2, 3, 4, 5, 6, 8, 10, 12, None],
              "clf__min_samples_leaf": [1, 5, 10, 20],
              "clf__class_weight": [None, "balanced"]},
        plot=("clf__max_depth", "clf__min_samples_leaf"),
    ),
    "svm_rbf": dict(
        estimator=SVC(kernel="rbf", probability=SVM_PROBABILITY, random_state=SEED),
        scale=True,
        grid={"clf__C": [0.1, 1, 10, 100],
              "clf__gamma": ["scale", 0.01, 0.1, 1],
              "clf__class_weight": [None, "balanced"]},
        plot=("clf__C", "clf__gamma"),
    ),
}


def tune(name, X_train, y_train):
    """Grid-search one model's pipeline with stratified CV; the best pipeline is refit on all of X_train."""
    spec = MODELS[name]
    pipeline = Pipeline([("prep", make_preprocessor(scale=spec["scale"])), ("clf", spec["estimator"])])
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    # return_train_score: also score each candidate on its training folds, to expose overfitting.
    search = GridSearchCV(pipeline, spec["grid"], scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=-1,
                          return_train_score=True)
    return search.fit(X_train, y_train)


def _sort_validation_group(group, x_param):
    """Sort numeric grid values numerically, leaving null candidates (for example None) last."""
    column = f"param_{x_param}"
    numeric = pd.to_numeric(group[column], errors="coerce")
    if numeric.notna().any():
        missing = numeric.isna().to_numpy()
        order = np.lexsort((numeric.fillna(np.inf).to_numpy(), missing))
        return group.iloc[order]
    return group.sort_values(column, key=lambda values: values.astype(str), kind="stable")


def _format_validation_tick(value):
    """Show null grid candidates as their Python parameter value instead of pandas' NaN."""
    return "None" if pd.isna(value) else str(value)


def plot_validation_curve(search, x_param, line_param, path):
    """Mean train and CV score vs one hyperparameter (one colour per value of another); other params fixed at their best.

    A large gap between the dashed train line and the solid CV line indicates overfitting.
    """
    res = pd.DataFrame(search.cv_results_)
    required = {f"param_{x_param}", f"param_{line_param}",
                f"mean_test_{REFIT_METRIC}", f"std_test_{REFIT_METRIC}", f"mean_train_{REFIT_METRIC}"}
    if not required.issubset(res.columns):
        raise ValueError(f"validation curve columns unavailable: {sorted(required - set(res.columns))}")
    for p, v in search.best_params_.items():
        if p not in (x_param, line_param):
            column = f"param_{p}"
            if column not in res:
                raise ValueError(f"validation curve parameter column unavailable: {column}")
            matches = res[column].isna() if pd.isna(v) else res[column].astype(str) == str(v)
            res = res[matches]
    if res.empty:
        raise ValueError("no validation curve results match the best fixed parameters")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = list(res.groupby(res[f"param_{line_param}"].astype(str), sort=False))
    if not groups:
        plt.close(fig)
        raise ValueError("no validation curve groups are available")
    for line_value, group in groups:
        group = _sort_validation_group(group, x_param)
        series = f"{line_param.removeprefix('clf__')}={line_value}"
        cv_line = ax.errorbar(range(len(group)), group[f"mean_test_{REFIT_METRIC}"],
                              yerr=group[f"std_test_{REFIT_METRIC}"], marker="o", capsize=3,
                              label=f"mean CV {REFIT_METRIC} ({series})")
        ax.plot(range(len(group)), group[f"mean_train_{REFIT_METRIC}"], marker="s", linestyle="--",
                color=cv_line[0].get_color(), label=f"mean train {REFIT_METRIC} ({series})")
    tick_group = _sort_validation_group(groups[0][1], x_param)
    tick_labels = tick_group[f"param_{x_param}"].map(_format_validation_tick)
    ax.set_xticks(range(len(tick_group)), tick_labels)
    ax.set_xlabel(x_param.removeprefix("clf__"))
    ax.set_ylabel(f"{REFIT_METRIC} over {CV_FOLDS} folds (CV ± std)")
    fixed = {p.removeprefix("clf__"): v for p, v in search.best_params_.items() if p not in (x_param, line_param)}
    ax.set_title(f"{path.stem.removesuffix('_validation_curve')}: validation curve" + (f"  {fixed}" if fixed else ""))
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_model(name, estimator, X_test, y_test, out_dir):
    """Evaluate a fitted estimator once on the held-out test set and save metrics and plots."""
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X_test)
        score_method = "decision_function"
    elif hasattr(estimator, "predict_proba"):
        positive_index = list(estimator.classes_).index(FAILURE_LABEL)
        y_score = estimator.predict_proba(X_test)[:, positive_index]
        score_method = "predict_proba"
    else:
        raise ValueError(f"{name} estimator provides neither decision_function nor predict_proba")

    metrics = {
        "model": name,
        "n_test": int(len(y_test)),
        "score_method": score_method,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=FAILURE_LABEL, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=FAILURE_LABEL, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=FAILURE_LABEL, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "average_precision": float(average_precision_score(y_test, y_score, pos_label=FAILURE_LABEL)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    matrix = confusion_matrix(y_test, y_pred, labels=[0, FAILURE_LABEL])
    pd.DataFrame(matrix, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(
        out_dir / f"{name}_confusion_matrix.csv", index=True, index_label="actual")

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=FAILURE_LABEL)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{name}: held-out ROC curve",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_roc_curve.png", dpi=120)
    plt.close(fig)
    return metrics


def save_outputs(name, search, X_train, X_test, y_train, y_test):
    model_dir, out_dir = ARTIFACTS_DIR / "classification", OUTPUTS_DIR / "classification"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(search.best_estimator_, model_dir / f"{name}.joblib")
    pd.DataFrame(search.cv_results_).to_csv(out_dir / f"{name}_cv_results.csv", index=False)
    try:
        plot_validation_curve(search, *MODELS[name]["plot"], out_dir / f"{name}_validation_curve.png")
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        print(f"could not plot {name} validation curve: {exc}")

    best = search.best_index_
    metadata = {
        "model": name,
        "pipeline_file": f"{name}.joblib",
        "target": TARGET,
        "features": {"categorical": CATEGORICAL, "numeric": NUMERIC},
        "excluded_columns": {"identifiers": ID_COLUMNS, "target_leakage": LEAKAGE_COLUMNS},
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_train": len(X_train),
        # Original DataFrame indices (CSV row positions) make the shared held-out split auditable.
        "rows": {"train": X_train.index.tolist(), "test": X_test.index.tolist()},
        "versions": {
            "python": platform.python_version(),
            **{package: _package_version(distribution) for package, distribution in {
                "numpy": "numpy", "pandas": "pandas", "scikit_learn": "scikit-learn",
                "joblib": "joblib", "tensorflow": "tensorflow", "keras": "keras",
            }.items()},
        },
        "train_failure_rate": float(y_train.mean()),
        "cv_folds": CV_FOLDS,
        "selection_metric": REFIT_METRIC,
        "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
        "best_cv_scores": {m: {"mean": float(search.cv_results_[f"mean_test_{m}"][best]),
                               "std": float(search.cv_results_[f"std_test_{m}"][best])} for m in SCORING},
    }
    (model_dir / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2))
    metadata["test_scores"] = evaluate_model(name, search.best_estimator_, X_test, y_test, out_dir)
    return metadata


def _package_version(distribution):
    """Return an installed distribution version, or None when the optional package is absent."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()

    df = load_dataset()
    describe_dataset(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"\nTrain: {len(X_train)} rows ({y_train.mean():.2%} failures) | "
          f"held-out test: {len(X_test)} rows ({y_test.mean():.2%} failures)")

    for name in args.models:
        print(f"\n=== {name}: searching {MODELS[name]['grid']} ===")
        meta = save_outputs(name, tune(name, X_train, y_train), X_train, X_test, y_train, y_test)
        print(f"best params: {meta['best_params']}")
        for m, s in meta["best_cv_scores"].items():
            print(f"  CV {m:<18} {s['mean']:.4f} ± {s['std']:.4f}")


if __name__ == "__main__":
    main()
