"""Checks numeric validation-curve ordering and null parameter labels."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from classification.train import (_format_validation_tick, _sort_validation_group,
                                  plot_validation_curve)


class TestValidationCurvePlotting(unittest.TestCase):
    def test_numeric_grid_values_sort_numerically_and_none_label_is_preserved(self):
        group = pd.DataFrame({"param_clf__max_depth": [12, 2, 10, None]})
        ordered = _sort_validation_group(group, "clf__max_depth")
        self.assertEqual(ordered["param_clf__max_depth"].iloc[:3].tolist(), [2.0, 10.0, 12.0])
        self.assertTrue(pd.isna(ordered["param_clf__max_depth"].iloc[3]))
        self.assertEqual(_format_validation_tick(ordered["param_clf__max_depth"].iloc[3]), "None")

    def test_plot_is_written_for_multiple_numeric_candidates(self):
        results = pd.DataFrame({
            "param_clf__n_neighbors": [11, 1, 3],
            "param_clf__weights": ["uniform"] * 3,
            "mean_test_f1": [0.3, 0.5, 0.4],
            "std_test_f1": [0.02] * 3,
            "mean_train_f1": [0.4, 1.0, 0.8],
        })
        search = SimpleNamespace(cv_results_=results,
                                 best_params_={"clf__n_neighbors": 1, "clf__weights": "uniform"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "knn_validation_curve.png"
            plot_validation_curve(search, "clf__n_neighbors", "clf__weights", path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
