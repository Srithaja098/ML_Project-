"""Synthetic checks for held-out evaluation outputs; no dataset download is needed."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classification.train import evaluate_model


class TestClassificationEvaluation(unittest.TestCase):
    def setUp(self):
        self.X_test = pd.DataFrame({"signal": [-2, -1, 1, 2, -0.5, 0.5]})
        self.y_test = pd.Series([0, 0, 1, 1, 0, 1])

    def test_outputs_and_metrics_for_probability_estimator(self):
        model = DecisionTreeClassifier(random_state=0).fit(self.X_test, self.y_test)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            metrics = evaluate_model("decision_tree", model, self.X_test, self.y_test, out_dir)
            self.assertEqual(metrics["n_test"], len(self.y_test))
            self.assertEqual(metrics["score_method"], "predict_proba")
            self.assertEqual(metrics["f1"], 1.0)
            self.assertEqual(set(json.loads((out_dir / "decision_tree_test_metrics.json").read_text())),
                             set(metrics))
            matrix = pd.read_csv(out_dir / "decision_tree_confusion_matrix.csv", index_col=0)
            np.testing.assert_array_equal(matrix.to_numpy(), [[3, 0], [0, 3]])
            self.assertGreater((out_dir / "decision_tree_roc_curve.png").stat().st_size, 0)

    def test_svm_uses_decision_function(self):
        model = SVC(kernel="rbf", probability=False, random_state=0).fit(self.X_test, self.y_test)
        with tempfile.TemporaryDirectory() as tmp:
            metrics = evaluate_model("svm_rbf", model, self.X_test, self.y_test, Path(tmp))
            self.assertEqual(metrics["score_method"], "decision_function")
            self.assertGreaterEqual(metrics["roc_auc"], 0.0)
            self.assertLessEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
