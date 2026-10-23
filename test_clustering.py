"""Unit tests for the clustering module on Palmer Penguins dataset."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from clustering.data import (
    NUMERIC_FEATURES,
    TARGET,
    clean_dataset,
    prepare_clustering_data,
    scale_features,
)
from clustering.cluster import (
    compute_contingency,
    compute_metrics,
    CLUSTERING_ARTIFACTS_DIR,
    CLUSTERING_OUTPUTS_DIR,
)


class TestClusteringData(unittest.TestCase):
    def setUp(self):
        # Create synthetic test dataframe resembling Palmer Penguins
        self.raw_df = pd.DataFrame({
            "species": ["Adelie", "Adelie", "Gentoo", "Chinstrap", "Adelie"],
            "bill_length_mm": [39.1, 39.5, 46.1, 50.0, np.nan],
            "bill_depth_mm": [18.7, 17.4, 13.2, 19.5, np.nan],
            "flipper_length_mm": [181.0, 186.0, 211.0, 196.0, np.nan],
            "body_mass_g": [3750.0, 3800.0, 4500.0, 3900.0, np.nan],
            "island": ["Torgersen", "Torgersen", "Biscoe", "Dream", "Torgersen"],
            "sex": ["male", "female", "female", "male", np.nan],
            "year": [2007, 2007, 2007, 2008, 2007],
        })

    def test_clean_dataset_removes_missing_numerics(self):
        cleaned = clean_dataset(self.raw_df)
        self.assertEqual(len(cleaned), 4)
        self.assertFalse(cleaned[NUMERIC_FEATURES].isna().any().any())

    def test_prepare_clustering_data_shapes(self):
        X, y, meta = prepare_clustering_data(self.raw_df)
        self.assertEqual(X.shape, (4, 4))
        self.assertEqual(len(y), 4)
        self.assertIn("island", meta.columns)

    def test_scale_features(self):
        X, _, _ = prepare_clustering_data(self.raw_df)
        X_scaled, scaler = scale_features(X)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertEqual(X_scaled.shape, (4, 4))
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1, atol=1e-7)


class TestClusteringMetrics(unittest.TestCase):
    def test_compute_metrics_perfect_clustering(self):
        X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
        labels = np.array([0, 0, 1, 1])
        y_true = pd.Series(["A", "A", "B", "B"])
        metrics = compute_metrics(X, labels, y_true)
        self.assertEqual(metrics["n_clusters"], 2)
        self.assertEqual(metrics["n_noise"], 0)
        self.assertEqual(metrics["adjusted_rand_index"], 1.0)
        self.assertEqual(metrics["v_measure"], 1.0)
        self.assertGreater(metrics["silhouette"], 0.8)

    def test_compute_contingency(self):
        labels = np.array([0, 0, 1, 1])
        y_true = pd.Series(["Adelie", "Adelie", "Gentoo", "Gentoo"])
        contingency = compute_contingency(labels, y_true)
        self.assertEqual(contingency.loc[0, "Adelie"], 2)
        self.assertEqual(contingency.loc[1, "Gentoo"], 2)


class TestClusteringArtifactsAndOutputs(unittest.TestCase):
    def test_artifacts_exist_and_non_empty(self):
        required_artifacts = [
            "scaler.joblib",
            "kmeans.joblib",
            "agglomerative.joblib",
            "dbscan.joblib",
            "gmm.joblib",
            "pca.joblib",
            "metadata.json",
        ]
        for name in required_artifacts:
            path = CLUSTERING_ARTIFACTS_DIR / name
            self.assertTrue(path.exists(), f"Missing artifact: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty artifact: {path}")

    def test_outputs_exist_and_non_empty(self):
        required_outputs = [
            "clustering_summary_metrics.csv",
            "kmeans_k_search_results.csv",
            "kmeans_contingency_matrix.csv",
            "agglomerative_contingency_matrix.csv",
            "dbscan_contingency_matrix.csv",
            "gmm_contingency_matrix.csv",
            "kmeans_elbow_silhouette.png",
            "hierarchical_dendrogram.png",
            "dbscan_k_distance.png",
            "gmm_bic_aic.png",
            "pca_clustering_comparison.png",
            "feature_pairwise_clusters.png",
        ]
        for name in required_outputs:
            path = CLUSTERING_OUTPUTS_DIR / name
            self.assertTrue(path.exists(), f"Missing output file: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty output file: {path}")


if __name__ == "__main__":
    unittest.main()
