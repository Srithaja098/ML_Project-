"""Unit tests for the regression module on Concrete Compressive Strength dataset."""
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from regression.data import (
    FEATURES,
    TARGET,
    describe_dataset,
    load_dataset,
    load_splits,
    make_preprocessor,
    split_features_target,
    train_test_split_dataset,
)
from regression.train import evaluate_predictions
from regression.assignment import OUT as ASSIGNMENT_REGRESSION_OUTPUTS


class TestRegressionData(unittest.TestCase):
    def setUp(self):
        # Synthetic data with 8 features + 1 target
        n = 100
        rng = np.random.default_rng(42)
        data = {f: rng.uniform(10, 500, n) for f in FEATURES}
        data[TARGET] = rng.uniform(5, 80, n)
        self.raw_df = pd.DataFrame(data)

    def test_split_features_target(self):
        X, y = split_features_target(self.raw_df)
        self.assertEqual(X.shape, (100, 8))
        self.assertEqual(len(y), 100)
        self.assertNotIn(TARGET, X.columns)

    def test_train_test_split_ratio(self):
        X, y = split_features_target(self.raw_df)
        X_tr, X_te, y_tr, y_te = train_test_split_dataset(X, y, seed=42, test_size=0.20)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        self.assertEqual(len(y_tr), 80)
        self.assertEqual(len(y_te), 20)

    def test_preprocessor_scaling(self):
        X, _ = split_features_target(self.raw_df)
        scaler = make_preprocessor()
        X_scaled = scaler.fit_transform(X)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertEqual(X_scaled.shape, (100, 8))
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1, atol=1e-7)


class TestRegressionMetrics(unittest.TestCase):
    def test_evaluate_predictions_perfect_fit(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertEqual(metrics["r2"], 1.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["medae"], 0.0)
        self.assertEqual(metrics["mape"], 0.0)

    def test_evaluate_predictions_known_deviation(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 32.0, 38.0])
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertAlmostEqual(metrics["rmse"], 2.0, places=3)
        self.assertAlmostEqual(metrics["mae"], 2.0, places=3)
        self.assertGreater(metrics["r2"], 0.9)


class TestRegressionArtifactsAndOutputs(unittest.TestCase):
    def test_assignment_metrics_and_selected_regularization_exist(self):
        path = ASSIGNMENT_REGRESSION_OUTPUTS / "metrics.json"
        self.assertTrue(path.exists(), f"Run python -m regression.assignment first: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(metrics["n_train"], 353)
        self.assertEqual(metrics["n_test"], 89)
        self.assertTrue(metrics["gradient_descent"]["converged"])
        self.assertIn("selected_alpha", metrics["models"]["Ridge"])
        self.assertIn("selected_alpha", metrics["models"]["Lasso"])

    def test_outputs_exist_and_non_empty(self):
        required_outputs = [
            "polynomial_complexity.csv",
            "actual_vs_predicted.png",
            "complexity_curve.png",
            "ridge_cv.csv",
            "lasso_cv.csv",
            "test_predictions.csv",
        ]
        for name in required_outputs:
            path = ASSIGNMENT_REGRESSION_OUTPUTS / name
            self.assertTrue(path.exists(), f"Missing output file: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty output file: {path}")


if __name__ == "__main__":
    unittest.main()
