"""Focused tests for the explicitly required from-scratch algorithms."""
import unittest

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from config import SEED, TEST_SIZE
from regression.assignment import GradientDescentLinearRegression, metrics
from clustering.scratch_kmeans import kmeans
from pca.scratch_pca import fit_pca


class AssignmentAlgorithmTests(unittest.TestCase):
    def test_gradient_descent_converges_on_linear_data(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(300, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + 4
        model = GradientDescentLinearRegression(learning_rate=0.1, max_iter=10000).fit(X, y)
        self.assertTrue(model.converged_)
        np.testing.assert_allclose(model.predict(X), y, atol=1e-4)
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_metrics_separate_r_from_r_squared(self):
        result = metrics([1, 2, 3, 4], [1, 2, 2, 4])
        self.assertAlmostEqual(result["pearson_r"], np.corrcoef([1, 2, 3, 4], [1, 2, 2, 4])[0, 1])
        self.assertIn("r2", result)
        self.assertIn("mse", result)
        self.assertIn("rmse", result)

    def test_train_test_split_is_eighty_twenty(self):
        data = load_diabetes()
        Xtr, Xte, _, _ = train_test_split(data.data, data.target, random_state=SEED, test_size=TEST_SIZE)
        self.assertEqual(len(Xtr), 353)
        self.assertEqual(len(Xte), 89)

    def test_numpy_kmeans_finite_and_requested_k_values(self):
        rng = np.random.default_rng(3)
        X = np.r_[rng.normal(-4, .2, (30, 2)), rng.normal(0, .2, (30, 2)), rng.normal(4, .2, (30, 2))]
        for k in range(2, 7):
            labels, centers, wcss, _ = kmeans(X, k, seed=SEED)
            self.assertTrue(np.isfinite(centers).all())
            self.assertTrue(np.isfinite(wcss))
            self.assertLessEqual(wcss, float(np.sum((X - X.mean(0)) ** 2)))
            self.assertEqual(labels.shape, (len(X),))

    def test_kmeans_handles_duplicate_points_and_empty_clusters(self):
        X = np.zeros((8, 2))
        labels, centers, wcss, _ = kmeans(X, 4, seed=SEED)
        self.assertTrue(np.isfinite(centers).all())
        self.assertEqual(wcss, 0.0)

    def test_pca_eigen_properties_and_dimensions(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40, 6))
        result = fit_pca(X)
        self.assertTrue(np.all(np.diff(result["eigenvalues"]) <= 1e-12))
        self.assertAlmostEqual(result["explained_variance_ratio"].sum(), 1.0)
        self.assertEqual(((X - X.mean(0)) @ result["eigenvectors"][:, :2]).shape, (40, 2))
        self.assertEqual(result["eigenvectors"][:, :3].shape, (6, 3))
        self.assertTrue(np.all(result["explained_variance_ratio"] >= 0))

    def test_pca_run_outputs_include_two_and_three_components(self):
        from config import OUTPUTS_DIR
        out = OUTPUTS_DIR / "assignment_pca"
        self.assertTrue((out / "diabetes_2_components.csv").is_file())
        self.assertTrue((out / "diabetes_3_components.csv").is_file())
        self.assertTrue((out / "cumulative_explained_variance.png").is_file())
        self.assertTrue((out / "pca_2d_visualization.png").is_file())
        self.assertTrue((out / "metadata.json").is_file())

    def test_combined_report_is_generated(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent
        for name in ["ML_Assignment_Report.md", "ML_Assignment_Report.pdf"]:
            path = root / "reports" / name
            self.assertTrue(path.is_file(), f"Missing report: {path}")
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
