"""Train and evaluate regression models on Concrete Compressive Strength dataset.

Evaluates Ridge Regression, K-Nearest Neighbors, Decision Tree, Random Forest,
Gradient Boosting, and Support Vector Regression (SVR-RBF) with 5-fold cross-validation
and held-out test evaluation (80:20 split, seed=42).

Usage:
    python -m regression.train
"""
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from config import ARTIFACTS_DIR, OUTPUTS_DIR, SEED, TEST_SIZE
from regression.data import (
    FEATURES,
    TARGET,
    describe_dataset,
    load_dataset,
    load_splits,
    make_preprocessor,
)

REGRESSION_OUTPUTS_DIR = OUTPUTS_DIR / "regression"
REGRESSION_ARTIFACTS_DIR = ARTIFACTS_DIR / "regression"
CV_FOLDS = 5

SCORING = {
    "neg_rmse": "neg_root_mean_squared_error",
    "neg_mae": "neg_mean_absolute_error",
    "r2": "r2",
}
REFIT_METRIC = "neg_rmse"

MODEL_SPECS = {
    "ridge": {
        "name": "Ridge Regression",
        "estimator": Ridge(random_state=SEED),
        "scale": True,
        "grid": {
            "reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 500.0],
        },
        "plot_param": "reg__alpha",
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "estimator": KNeighborsRegressor(),
        "scale": True,
        "grid": {
            "reg__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "reg__weights": ["uniform", "distance"],
        },
        "plot_param": "reg__n_neighbors",
    },
    "decision_tree": {
        "name": "Decision Tree",
        "estimator": DecisionTreeRegressor(random_state=SEED),
        "scale": False,
        "grid": {
            "reg__max_depth": [3, 5, 7, 10, 15, None],
            "reg__min_samples_leaf": [1, 2, 4, 8],
        },
        "plot_param": "reg__max_depth",
    },
    "random_forest": {
        "name": "Random Forest",
        "estimator": RandomForestRegressor(random_state=SEED),
        "scale": False,
        "grid": {
            "reg__n_estimators": [50, 100, 200],
            "reg__max_depth": [8, 12, 16, None],
            "reg__min_samples_leaf": [1, 2],
        },
        "plot_param": "reg__max_depth",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "estimator": GradientBoostingRegressor(random_state=SEED),
        "scale": False,
        "grid": {
            "reg__n_estimators": [100, 200, 300],
            "reg__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "reg__max_depth": [3, 4, 5],
        },
        "plot_param": "reg__learning_rate",
    },
    "svm_rbf": {
        "name": "Support Vector Regression (RBF)",
        "estimator": SVR(kernel="rbf"),
        "scale": True,
        "grid": {
            "reg__C": [1.0, 10.0, 50.0, 100.0, 200.0],
            "reg__epsilon": [0.05, 0.1, 0.2],
            "reg__gamma": ["scale", "auto", 0.05, 0.1],
        },
        "plot_param": "reg__C",
    },
}


def ensure_directories():
    """Create output and artifact directories."""
    REGRESSION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REGRESSION_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline(spec: dict) -> Pipeline:
    """Build sklearn Pipeline, including StandardScaler when scale=True."""
    steps = []
    if spec["scale"]:
        steps.append(("scaler", make_preprocessor()))
    steps.append(("reg", spec["estimator"]))
    return Pipeline(steps)


def tune_and_fit(key: str, spec: dict, X_train: pd.DataFrame, y_train: pd.Series):
    """Perform 5-fold cross-validated grid search and refit best estimator."""
    pipeline = build_pipeline(spec)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        pipeline,
        param_grid=spec["grid"],
        scoring=SCORING,
        refit=REFIT_METRIC,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    """Compute standard regression metrics on held-out test data."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "medae": round(medae, 4),
        "mape": round(mape, 4),
    }


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

def plot_actual_vs_predicted(results: dict, y_test: pd.Series):
    """Plot Actual vs. Predicted values with y = x identity line for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    min_val = float(min(y_test.min(), min(r["y_pred"].min() for r in results.values())) - 2)
    max_val = float(max(y_test.max(), max(r["y_pred"].max() for r in results.values())) + 2)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (key, res) in enumerate(results.items()):
        ax = axes[idx]
        y_pred = res["y_pred"]
        test_m = res["test_metrics"]
        ax.scatter(y_test, y_pred, alpha=0.65, color=colors[idx], edgecolors="k", s=40)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.8, label="Ideal (y = x)")
        ax.set_title(f"{res['name']}\nR² = {test_m['r2']:.4f} | RMSE = {test_m['rmse']:.2f} MPa",
                     fontsize=11, weight="bold", color="#17324d")
        ax.set_xlabel("Actual Strength (MPa)", fontsize=10)
        ax.set_ylabel("Predicted Strength (MPa)", fontsize=10)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(REGRESSION_OUTPUTS_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)


def plot_residuals(results: dict, y_test: pd.Series):
    """Plot Residuals (Actual - Predicted) vs. Predicted values to check homoscedasticity."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (key, res) in enumerate(results.items()):
        ax = axes[idx]
        y_pred = res["y_pred"]
        residuals = y_test.values - y_pred
        ax.scatter(y_pred, residuals, alpha=0.65, color=colors[idx], edgecolors="k", s=40)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.8)
        ax.set_title(f"{res['name']} Residuals\nMean: {residuals.mean():.2f} | Std: {residuals.std():.2f}",
                     fontsize=11, weight="bold", color="#17324d")
        ax.set_xlabel("Predicted Strength (MPa)", fontsize=10)
        ax.set_ylabel("Residual (Actual - Pred)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(REGRESSION_OUTPUTS_DIR / "residuals_plot.png", dpi=200)
    plt.close(fig)


def plot_feature_importances(results: dict):
    """Plot tree-based feature importances (Gradient Boosting and Random Forest)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    # Gradient Boosting Importances
    gb_model = results["gradient_boosting"]["search"].best_estimator_.named_steps["reg"]
    gb_imp = pd.Series(gb_model.feature_importances_, index=FEATURES).sort_values()
    ax1.barh(gb_imp.index, gb_imp.values, color="#2ca02c", edgecolor="k", alpha=0.85)
    ax1.set_title("Gradient Boosting Feature Importance", fontsize=12, weight="bold", color="#17324d")
    ax1.set_xlabel("Gini / Gain Importance", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Random Forest Importances
    rf_model = results["random_forest"]["search"].best_estimator_.named_steps["reg"]
    rf_imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
    ax2.barh(rf_imp.index, rf_imp.values, color="#d62728", edgecolor="k", alpha=0.85)
    ax2.set_title("Random Forest Feature Importance", fontsize=12, weight="bold", color="#17324d")
    ax2.set_xlabel("MDI Feature Importance", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(REGRESSION_OUTPUTS_DIR / "feature_importance.png", dpi=200)
    plt.close(fig)


def plot_cv_model_comparison(results: dict):
    """Plot bar chart comparing cross-validation RMSE and R2 scores across all models."""
    names = [r["name"] for r in results.values()]
    cv_rmse = [abs(r["search"].cv_results_["mean_test_neg_rmse"][r["search"].best_index_]) for r in results.values()]
    cv_rmse_std = [r["search"].cv_results_["std_test_neg_rmse"][r["search"].best_index_] for r in results.values()]
    cv_r2 = [r["search"].cv_results_["mean_test_r2"][r["search"].best_index_] for r in results.values()]
    cv_r2_std = [r["search"].cv_results_["std_test_r2"][r["search"].best_index_] for r in results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(names))

    # RMSE bar chart (lower is better)
    bars1 = ax1.bar(x, cv_rmse, yerr=cv_rmse_std, capsize=5, color="#1f77b4", edgecolor="k", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    ax1.set_title("5-Fold CV Root Mean Squared Error (RMSE, Lower is Better)", fontsize=11.5, weight="bold", color="#17324d")
    ax1.set_ylabel("CV RMSE (MPa)", fontsize=10.5)
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{yval:.2f}", ha="center", fontsize=9)

    # R2 bar chart (higher is better)
    bars2 = ax2.bar(x, cv_r2, yerr=cv_r2_std, capsize=5, color="#2ca02c", edgecolor="k", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    ax2.set_title("5-Fold CV Coefficient of Determination (R², Higher is Better)", fontsize=11.5, weight="bold", color="#17324d")
    ax2.set_ylabel("CV R² Score", fontsize=10.5)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(REGRESSION_OUTPUTS_DIR / "cv_model_comparison.png", dpi=200)
    plt.close(fig)


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def run_regression_pipeline():
    """Run full training, evaluation, plotting, and serialization workflow."""
    ensure_directories()
    print("Loading Concrete Compressive Strength dataset...")
    df = load_dataset()
    describe_dataset(df)

    X_train, X_test, y_train, y_test = load_splits()
    print(f"\nSplit dataset into {len(X_train)} training and {len(X_test)} held-out test rows (80:20).")

    results = {}
    summary_rows = []

    for key, spec in MODEL_SPECS.items():
        print(f"\n---> Tuning and training {spec['name']} ({key})...")
        search = tune_and_fit(key, spec, X_train, y_train)
        best_est = search.best_estimator_
        best_idx = search.best_index_

        cv_rmse = abs(search.cv_results_["mean_test_neg_rmse"][best_idx])
        cv_rmse_std = search.cv_results_["std_test_neg_rmse"][best_idx]
        cv_r2 = search.cv_results_["mean_test_r2"][best_idx]
        cv_r2_std = search.cv_results_["std_test_r2"][best_idx]
        cv_mae = abs(search.cv_results_["mean_test_neg_mae"][best_idx])

        # Held-out test evaluation
        y_pred = best_est.predict(X_test)
        test_metrics = evaluate_predictions(y_test.values, y_pred)

        results[key] = {
            "name": spec["name"],
            "search": search,
            "best_estimator": best_est,
            "best_params": search.best_params_,
            "cv_metrics": {
                "cv_rmse": round(float(cv_rmse), 4),
                "cv_rmse_std": round(float(cv_rmse_std), 4),
                "cv_r2": round(float(cv_r2), 4),
                "cv_r2_std": round(float(cv_r2_std), 4),
                "cv_mae": round(float(cv_mae), 4),
            },
            "test_metrics": test_metrics,
            "y_pred": y_pred,
        }

        # Save model artifact
        joblib.dump(best_est, REGRESSION_ARTIFACTS_DIR / f"{key}.joblib")

        # Save individual CV results table
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(REGRESSION_OUTPUTS_DIR / f"{key}_cv_results.csv", index=False)

        # Save individual test metrics JSON
        with open(REGRESSION_OUTPUTS_DIR / f"{key}_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "model": spec["name"],
                "best_params": search.best_params_,
                "cv_metrics": results[key]["cv_metrics"],
                "test_metrics": test_metrics,
            }, f, indent=2)

        summary_rows.append({
            "Model": spec["name"],
            "Best_Hyperparameters": str(search.best_params_),
            "CV_RMSE (+/- std)": f"{cv_rmse:.3f} (+/-{cv_rmse_std:.3f})",
            "CV_R2 (+/- std)": f"{cv_r2:.3f} (+/-{cv_r2_std:.3f})",
            "Test_R2": test_metrics["r2"],
            "Test_RMSE": test_metrics["rmse"],
            "Test_MAE": test_metrics["mae"],
            "Test_MedAE": test_metrics["medae"],
            "Test_MAPE": f"{test_metrics['mape'] * 100:.2f}%",
        })

    # Summary table
    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by="Test_R2", ascending=False, inplace=True)
    df_summary.to_csv(REGRESSION_OUTPUTS_DIR / "regression_summary_metrics.csv", index=False)

    # Save fitted preprocessor
    scaler = make_preprocessor().fit(X_train)
    joblib.dump(scaler, REGRESSION_ARTIFACTS_DIR / "scaler.joblib")

    # Generate all plots
    print("\nGenerating diagnostic plots...")
    plot_actual_vs_predicted(results, y_test)
    plot_residuals(results, y_test)
    plot_feature_importances(results)
    plot_cv_model_comparison(results)

    # Save metadata JSON
    metadata = {
        "dataset": "Concrete Compressive Strength (UCI ID 165)",
        "source_url": "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": FEATURES,
        "target": TARGET,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "models": {k: {"best_params": v["best_params"], "cv": v["cv_metrics"], "test": v["test_metrics"]}
                   for k, v in results.items()},
    }
    with open(REGRESSION_ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 90)
    print("CONCRETE REGRESSION EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)
    print(f"Summary metrics saved to: {REGRESSION_OUTPUTS_DIR / 'regression_summary_metrics.csv'}")
    print(f"Artifacts saved to: {REGRESSION_ARTIFACTS_DIR}")
    return df_summary


if __name__ == "__main__":
    run_regression_pipeline()
