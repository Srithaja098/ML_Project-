"""Unit tests for the regression module on Concrete Compressive Strength dataset."""
import unittest
from pathlib import Path
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
from regression.train import (
    REGRESSION_ARTIFACTS_DIR,
    REGRESSION_OUTPUTS_DIR,
    evaluate_predictions,
)


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
    def test_artifacts_exist_and_non_empty(self):
        required_artifacts = [
            "ridge.joblib",
            "knn.joblib",
            "decision_tree.joblib",
            "random_forest.joblib",
            "gradient_boosting.joblib",
            "svm_rbf.joblib",
            "scaler.joblib",
            "metadata.json",
        ]
        for name in required_artifacts:
            path = REGRESSION_ARTIFACTS_DIR / name
            self.assertTrue(path.exists(), f"Missing artifact: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty artifact: {path}")

    def test_outputs_exist_and_non_empty(self):
        required_outputs = [
            "regression_summary_metrics.csv",
            "actual_vs_predicted.png",
            "residuals_plot.png",
            "feature_importance.png",
            "cv_model_comparison.png",
            "ridge_cv_results.csv",
            "ridge_test_metrics.json",
            "random_forest_cv_results.csv",
            "gradient_boosting_cv_results.csv",
            "svm_rbf_cv_results.csv",
        ]
        for name in required_outputs:
            path = REGRESSION_OUTPUTS_DIR / name
            self.assertTrue(path.exists(), f"Missing output file: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty output file: {path}")


if __name__ == "__main__":
    unittest.main()
