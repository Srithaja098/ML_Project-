"""Synthetic tests for CNN held-out evaluation and saved split-index checks."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import keras
    from cnn.evaluate import CLASSICAL_MODELS, save_test_evaluation, verify_test_indices
    from cnn.model import build_cnn
    HAS_KERAS = True
except ImportError:
    HAS_KERAS = False


@unittest.skipUnless(HAS_KERAS, "keras/tensorflow not installed")
class TestCnnEvaluation(unittest.TestCase):
    def test_cnn_forward_train_and_evaluate_on_synthetic_batch(self):
        keras.utils.set_random_seed(42)
        model = build_cnn(8, conv_filters=(4,), dense_units=4, dropout=0.0)
        X = np.random.default_rng(42).normal(size=(16, 8, 1)).astype("float32")
        y = np.array([0, 1] * 8, dtype="float32")
        prediction = model(X, training=False).numpy()
        self.assertEqual(prediction.shape, (16, 1))
        self.assertTrue(np.isfinite(prediction).all())
        model.fit(X, y, epochs=1, batch_size=4, verbose=0)
        evaluation = model.evaluate(X, y, verbose=0, return_dict=True)
        self.assertTrue(np.isfinite(evaluation["loss"]))

    def test_test_indices_must_match_all_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cnn_metadata = root / "cnn_metadata.json"
            cnn_metadata.write_text(json.dumps({"rows": {"test": [4, 8, 9]}}))
            classical = root / "classical"
            classical.mkdir()
            for name in CLASSICAL_MODELS:
                (classical / f"{name}_metadata.json").write_text(
                    json.dumps({"rows": {"test": [4, 8, 9]}}))
            self.assertEqual(verify_test_indices([4, 8, 9], cnn_metadata, classical), 3)
            (classical / "svm_rbf_metadata.json").write_text(
                json.dumps({"rows": {"test": [4, 8, 10]}}))
            with self.assertRaisesRegex(ValueError, "do not match svm_rbf"):
                verify_test_indices([4, 8, 9], cnn_metadata, classical)

    def test_metrics_and_files_use_probabilities_and_fixed_threshold(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.7, 0.9])
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            metrics = save_test_evaluation(labels, probabilities, out_dir)
            self.assertEqual(metrics["prediction_threshold"], 0.5)
            self.assertEqual(metrics["score_type"], "sigmoid_probability")
            self.assertEqual(metrics["accuracy"], 0.75)
            self.assertAlmostEqual(metrics["roc_auc"], 0.75)
            self.assertAlmostEqual(metrics["average_precision"], 0.8333333333333333)
            stored = json.loads((out_dir / "cnn_test_metrics.json").read_text())
            self.assertEqual(stored, metrics)
            matrix = pd.read_csv(out_dir / "cnn_confusion_matrix.csv", index_col=0)
            np.testing.assert_array_equal(matrix.to_numpy(), [[1, 1], [0, 2]])
            self.assertGreater((out_dir / "cnn_roc_curve.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
