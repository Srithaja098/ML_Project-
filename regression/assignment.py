"""Required regression assignment: GD linear, polynomial, Ridge, and Lasso.

Uses scikit-learn's bundled diabetes dataset (10 numeric measurements, 442 rows),
so runs without a network download. All standardization lives in training-only
pipelines. Run with ``python -m regression.assignment``.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import OUTPUTS_DIR, SEED, TEST_SIZE

OUT = OUTPUTS_DIR / "assignment_regression"


class GradientDescentLinearRegression:
    """Minimize mean squared error J(w,b)=mean((Xw+b-y)^2)/2 by batch GD."""
    def __init__(self, learning_rate=0.1, max_iter=20000, tolerance=1e-10, patience=20):
        self.learning_rate, self.max_iter = learning_rate, max_iter
        self.tolerance, self.patience = tolerance, patience

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1
        Z = (X - self.mean_) / self.scale_
        self.coef_ = np.zeros(Z.shape[1]); self.intercept_ = float(y.mean())
        self.loss_history_ = []; stable = 0
        for _ in range(self.max_iter):
            residual = Z @ self.coef_ + self.intercept_ - y
            loss = float(np.mean(residual ** 2) / 2)
            self.loss_history_.append(loss)
            grad_w = Z.T @ residual / len(y)
            grad_b = float(residual.mean())
            self.coef_ -= self.learning_rate * grad_w
            self.intercept_ -= self.learning_rate * grad_b
            if not np.isfinite(loss):
                raise FloatingPointError("Gradient descent diverged; reduce learning_rate")
            if len(self.loss_history_) > 1 and abs(self.loss_history_[-2] - loss) < self.tolerance:
                stable += 1
                if stable >= self.patience:
                    break
            else:
                stable = 0
        self.n_iter_ = len(self.loss_history_)
        if self.n_iter_ == self.max_iter and stable < self.patience:
            self.converged_ = False
        else:
            self.converged_ = True
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mean_) / self.scale_
        return Z @ self.coef_ + self.intercept_


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {"mse": float(mean_squared_error(y, pred)),
            "rmse": float(mean_squared_error(y, pred) ** 0.5),
            "pearson_r": float(np.corrcoef(y, pred)[0, 1]),
            "r2": float(r2_score(y, pred))}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = load_diabetes(as_frame=True)
    X, y = dataset.data, dataset.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "Gradient descent linear": GradientDescentLinearRegression(),
        "Polynomial degree 2": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge()),
        "Lasso": make_pipeline(StandardScaler(), Lasso(max_iter=50000)),
    }
    # Degree-1/2/3/5 complexity curve is selected using training-only CV.
    complexity = []
    for degree in [0, 1, 2, 3, 5]:
        if degree == 0:
            pipe = DummyRegressor(strategy="mean")
        else:
            pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False), StandardScaler(), LinearRegression())
        scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
        pipe.fit(Xtr, ytr)
        complexity.append({"degree": degree, "train_mse": float(mean_squared_error(ytr, pipe.predict(Xtr))),
                           "cv_mse": float(-scores.mean()), "cv_mse_std": float(scores.std())})
    pd.DataFrame(complexity).to_csv(OUT / "polynomial_complexity.csv", index=False)

    cv_records = {}
    for name, estimator in [("Ridge", make_pipeline(StandardScaler(), Ridge())),
                            ("Lasso", make_pipeline(StandardScaler(), Lasso(max_iter=50000)))]:
        rows = []
        for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
            estimator.set_params(**{f"{name.lower()}__alpha": alpha})
            score = cross_val_score(estimator, Xtr, ytr, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
            rows.append({"alpha": alpha, "cv_mse": float(-score.mean()), "cv_mse_std": float(score.std())})
        cv_records[name] = rows
        pd.DataFrame(rows).to_csv(OUT / f"{name.lower()}_cv.csv", index=False)
        chosen = min(rows, key=lambda row: row["cv_mse"])["alpha"]
        estimator.set_params(**{f"{name.lower()}__alpha": chosen})
        models[name] = estimator

    results = {}
    pred_table = pd.DataFrame({"actual": yte.to_numpy()})
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        results[name] = metrics(yte, pred)
        if name in cv_records:
            results[name]["selected_alpha"] = min(cv_records[name], key=lambda row: row["cv_mse"])["alpha"]
        pred_table[name] = pred
    pred_table.to_csv(OUT / "test_predictions.csv", index=False)
    gd = models["Gradient descent linear"]
    (OUT / "metrics.json").write_text(json.dumps({"dataset": "scikit-learn diabetes", "n_samples": len(X),
        "features": list(X.columns), "target": "disease progression one year after baseline",
        "seed": SEED, "test_size": TEST_SIZE, "n_train": len(Xtr), "n_test": len(Xte),
        "gradient_descent": {"loss": "mean((Xw + b - y)^2) / 2", "gradient_weights": "X.T @ residual / n",
            "gradient_intercept": "mean(residual)", "learning_rate": gd.learning_rate,
            "max_iter": gd.max_iter, "tolerance": gd.tolerance, "patience": gd.patience,
            "iterations": gd.n_iter_, "converged": gd.converged_,
            "initial_loss": gd.loss_history_[0], "final_loss": gd.loss_history_[-1]},
        "regularization_cv": cv_records, "models": results}, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in pred_table.drop(columns="actual").items():
        ax.scatter(pred_table.actual, p, s=18, alpha=.55, label=name)
    bounds = [float(yte.min()), float(yte.max())]
    ax.plot(bounds, bounds, "k--", lw=1); ax.set(xlabel="Actual progression", ylabel="Predicted progression", title="Held-out predictions")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "actual_vs_predicted.png", dpi=180); plt.close(fig)
    comp = pd.DataFrame(complexity)
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(comp.degree, comp.train_mse, "o-", label="Training MSE"); ax.plot(comp.degree, comp.cv_mse, "o-", label="5-fold validation MSE")
    ax.set(xlabel="Polynomial degree", ylabel="MSE", title="Polynomial complexity and generalization"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / "complexity_curve.png", dpi=180); plt.close(fig)
    print(pd.DataFrame(results).T.to_string())
    return results


if __name__ == "__main__":
    run()
