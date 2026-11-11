import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.csv_loader import CSVOptions
from core.dtypes import ColumnType
from core.metrics import Metric
from core.session import SessionError, load_session, save_session
from core.splitter import SplitCandidate, best_split
from core.tree import DecisionTreeModel, SplitNotAllowed


def make_model(n=200):
    rng = np.random.default_rng(5)
    x = rng.random(n)
    df = pd.DataFrame({"x": x, "y": (x < 0.5).astype(int)})
    types = {"x": ColumnType.NUMERIC, "y": ColumnType.NUMERIC}
    return DecisionTreeModel(df, types, "y", ["x"], Metric.GINI,
                             min_leaf=10, max_depth=3)


class TestTree(unittest.TestCase):
    def test_root_created(self):
        m = make_model()
        self.assertEqual(m.root.n_rows, 200)
        self.assertEqual(m.root.depth, 0)
        self.assertIn("metric_value", m.root.summary)

    def test_single_class_warning_not_blocker(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1, 1, 1, 1]})
        m = DecisionTreeModel(df, {"x": ColumnType.NUMERIC}, "y", ["x"],
                              Metric.GINI, min_leaf=1, max_depth=2)
        self.assertTrue(m.warnings)  # warning present but model exists

    def test_split_and_rules(self):
        m = make_model()
        cand = best_split(m.get_data(m.root.id), ["x"],
                          {"x": ColumnType.NUMERIC}, "y", Metric.GINI, 10)
        kids = m.split_node(m.root.id, cand)
        self.assertEqual(len(kids), 2)
        self.assertIn("<=", kids[0].rule)
        self.assertIn(">", kids[1].rule)
        self.assertEqual(kids[0].depth, 1)

    def test_max_depth_message(self):
        m = make_model()
        m.max_depth = 0
        cand = SplitCandidate("x", "numeric_cutoff", cutoff=0.5)
        with self.assertRaises(SplitNotAllowed) as ctx:
            m.split_node(m.root.id, cand)
        self.assertIn("Max depth", str(ctx.exception))

    def test_min_leaf_message(self):
        m = make_model()
        cand = SplitCandidate("x", "numeric_cutoff", cutoff=0.999)
        with self.assertRaises(SplitNotAllowed) as ctx:
            m.split_node(m.root.id, cand)
        self.assertIn("min leaf", str(ctx.exception))

    def test_missing_target(self):
        df = pd.DataFrame({"x": [1.0]})
        with self.assertRaises(SplitNotAllowed):
            DecisionTreeModel(df, {"x": ColumnType.NUMERIC}, "y", ["x"],
                              Metric.GINI)


class TestSession(unittest.TestCase):
    def test_roundtrip(self):
        m = make_model()
        cand = best_split(m.get_data(m.root.id), ["x"],
                          {"x": ColumnType.NUMERIC}, "y", Metric.GINI, 10)
        m.split_node(m.root.id, cand)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.tree"
            save_session(m, p, CSVOptions(), "orig.csv", ["log line 1"])
            m2, opts, src, logs = load_session(p)
        self.assertEqual(m2.root.n_rows, 200)
        self.assertEqual(len(m2.root.children), 2)
        self.assertEqual(m2.target, "y")
        self.assertEqual(src, "orig.csv")
        self.assertEqual(logs, ["log line 1"])
        # rebuilt data mapping consistent
        kid = next(iter(m2.root.children.values()))
        self.assertEqual(m2.get_data(kid.id).shape[0], kid.n_rows)

    def test_corrupt_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.tree"
            p.write_text("not a zip")
            with self.assertRaises(SessionError):
                load_session(p)


if __name__ == "__main__":
    unittest.main()
