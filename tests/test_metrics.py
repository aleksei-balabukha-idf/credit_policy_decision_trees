import unittest

import numpy as np
import pandas as pd

from core.metrics import Metric, entropy, gini, impurity, variance, node_summary


class TestMetrics(unittest.TestCase):
    def test_gini_pure(self):
        self.assertAlmostEqual(gini(np.array([1, 1, 1])), 0.0)

    def test_gini_balanced(self):
        self.assertAlmostEqual(gini(np.array([0, 0, 1, 1])), 0.5)

    def test_entropy_balanced(self):
        self.assertAlmostEqual(entropy(np.array(["a", "a", "b", "b"])), 1.0)

    def test_variance(self):
        self.assertAlmostEqual(variance(np.array([1.0, 2.0, 3.0])), np.var([1, 2, 3]))

    def test_impurity_ignores_nan(self):
        self.assertAlmostEqual(gini(np.array([0, 0, np.nan], dtype=object)), 0.0)

    def test_node_summary_classification(self):
        s = node_summary(pd.Series(["a", "a", "b", None], dtype="string"), Metric.GINI)
        self.assertEqual(s["n_rows"], 4)
        self.assertEqual(s["n_nan_target"], 1)
        self.assertEqual(s["prediction"], "a")
        self.assertEqual(s["distribution"], {"a": 2, "b": 1})

    def test_node_summary_regression(self):
        s = node_summary(pd.Series([1.0, 3.0]), Metric.MSE)
        self.assertEqual(s["prediction"], 2.0)
        self.assertEqual(s["n_rows"], 2)


if __name__ == "__main__":
    unittest.main()
