import unittest

import numpy as np
import pandas as pd

from core.dtypes import ColumnType
from core.metrics import Metric
from core.splitter import (CancelledError, SplitCandidate, apply_split,
                           best_split, split_masks)


def make_separable(n=200, cut=0.5):
    rng = np.random.default_rng(42)
    x = rng.random(n)
    y = (x < cut).astype(int)  # perfect separation at 0.5
    return pd.DataFrame({"x": x, "noise": rng.random(n), "y": y})


TYPES = {"x": ColumnType.NUMERIC, "noise": ColumnType.NUMERIC}


class TestSplitter(unittest.TestCase):
    def test_recovers_known_cutoff(self):
        df = make_separable()
        cand = best_split(df, ["x", "noise"], TYPES, "y",
                          Metric.GINI, min_leaf=10)
        self.assertIsNotNone(cand)
        self.assertEqual(cand.feature, "x")
        self.assertAlmostEqual(cand.cutoff, 0.5, delta=0.02)
        self.assertGreater(cand.gain_value, 0.4)

    def test_blocked_feature_not_used(self):
        df = make_separable()
        cand = best_split(df, ["x", "noise"], TYPES, "y",
                          Metric.GINI, min_leaf=10, blocked={"x"})
        if cand is not None:
            self.assertNotEqual(cand.feature, "x")

    def test_min_leaf_blocks_split(self):
        df = make_separable(n=40)
        cand = best_split(df, ["x"], TYPES, "y", Metric.GINI, min_leaf=30)
        self.assertIsNone(cand)  # 40 rows can't give two 30-row leaves

    def test_categorical_subset(self):
        rng = np.random.default_rng(1)
        n = 300
        cat = rng.choice(["good", "bad", "mid"], n)
        y = pd.Series(cat).isin(["bad"]).astype(int).to_numpy()
        df = pd.DataFrame({"c": pd.array(cat, dtype="string"), "y": y})
        cand = best_split(df, ["c"], {"c": ColumnType.CATEGORICAL}, "y",
                          Metric.GINI, min_leaf=10)
        self.assertEqual(cand.kind, "categorical_subset")
        self.assertEqual(cand.categories, ["bad"])
        self.assertGreater(cand.gain_value, 0.4)

    def test_regression_split(self):
        x = np.arange(200, dtype=float)
        y = np.where(x < 100, 0.0, 10.0)
        df = pd.DataFrame({"x": x, "y": y})
        cand = best_split(df, ["x"], TYPES, "y", Metric.MSE, min_leaf=10)
        self.assertAlmostEqual(cand.cutoff, 99.5, delta=1.0)
        self.assertGreater(cand.gain_value, 20.0)

    def test_nan_routing_left_right(self):
        df = make_separable()
        df.loc[df.sample(20, random_state=0).index, "x"] = np.nan
        cand = best_split(df, ["x"], TYPES, "y", Metric.GINI, min_leaf=5)
        self.assertIn(cand.nan_route, ("left", "right", "ignore", "separate"))
        self.assertEqual(cand.nan_rows, 20)

    def test_nan_as_category(self):
        n = 200
        x = np.random.default_rng(3).random(n)
        y = np.zeros(n, dtype=int)
        nan_idx = np.arange(n) % 4 == 0
        y[nan_idx] = 1
        df = pd.DataFrame({"x": np.where(nan_idx, np.nan, x), "y": y})
        cand = best_split(df, ["x"], TYPES, "y", Metric.GINI, min_leaf=10,
                          nan_as_category=True)
        self.assertEqual(cand.kind, "nan_split")
        m = split_masks(df["x"], cand)
        self.assertTrue(df.loc[m["left"], "y"].eq(1).all())

    def test_apply_split_separate_nan(self):
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0], "y": [0, 0, 1, 1]})
        cand = SplitCandidate("x", "numeric_cutoff", cutoff=2.5,
                              nan_route="separate")
        res = apply_split(df, cand)
        self.assertEqual(len(res["left"]), 2)
        self.assertEqual(len(res["right"]), 1)
        self.assertEqual(len(res["separate"]), 1)
        self.assertEqual(res["nan_rows"], 1)

    def test_cancellation(self):
        df = make_separable()
        with self.assertRaises(CancelledError):
            best_split(df, ["x"], TYPES, "y", Metric.GINI, 10,
                       cancel_cb=lambda: True)

    def test_progress_callback(self):
        df = make_separable()
        seen = []
        best_split(df, ["x", "noise"], TYPES, "y", Metric.GINI, 10,
                   progress_cb=lambda i, n: seen.append((i, n)))
        self.assertEqual(seen[-1], (2, 2))

    def test_performance_100k(self):
        import time
        rng = np.random.default_rng(7)
        n = 100_000
        data = {f"f{i}": rng.random(n) for i in range(10)}
        data["y"] = (data["f0"] < 0.5).astype(int)
        df = pd.DataFrame(data)
        types = {f"f{i}": ColumnType.NUMERIC for i in range(10)}
        t0 = time.perf_counter()
        cand = best_split(df, list(types), types, "y", Metric.GINI, 50)
        dt = time.perf_counter() - t0
        self.assertEqual(cand.feature, "f0")
        self.assertLess(dt, 10.0, f"split_best too slow: {dt:.1f}s")
        print(f"\n100k x 10 numeric split_best: {dt:.2f}s")


if __name__ == "__main__":
    unittest.main()
